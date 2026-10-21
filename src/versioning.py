"""知识库版本管理：按学年进行增量导入、重建、回滚。

核心能力：
- list_academic_years(): 列出当前库中所有学年
- count_by_academic_year(year): 统计某学年的 chunk 数量
- delete_by_academic_year(year): 删除某学年的全部 chunk（回滚）
- import_files(dir_path, academic_year): 增量导入新文件
- rebuild_academic_year(academic_year, dir_path): 重建某学年（先删后导）

设计原则：
- 所有写操作同时更新 Milvus 向量库和 data/processed/chunks.json（BM25 语料）
- 操作前打印影响范围，需确认后才执行（可通过 force=True 跳过）
"""
import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_milvus import Milvus

from src.preprocessing.pipeline import PreprocessingPipeline
from src.vectorstore import (
    COLLECTION_NAME,
    MILVUS_URI,
    build_embeddings,
    build_vector_store,
    load_vector_store,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------
def _get_milvus_client():
    """获取底层 Milvus 客户端（langchain 封装在 _milvus_client 中）。"""
    vs = load_vector_store()
    return vs._milvus_client


def list_academic_years(collection_name: str = COLLECTION_NAME) -> list[dict]:
    """列出当前库中所有学年及其 chunk 数量。

    返回：[{"academic_year": "2024-2025", "count": 120}, ...]
    """
    client = _get_milvus_client()
    # 查询全部数据，只取 academic_year 字段
    results = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["academic_year", "source_file"],
    )

    # 按学年聚合统计
    year_map: dict[str, int] = {}
    for r in results:
        year = r.get("academic_year", "") or "(未标记)"
        year_map[year] = year_map.get(year, 0) + 1

    return sorted(
        [{"academic_year": y, "count": c} for y, c in year_map.items()],
        key=lambda x: x["academic_year"],
    )


def count_by_academic_year(
    academic_year: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """统计指定学年的 chunk 数量。"""
    client = _get_milvus_client()
    results = client.query(
        collection_name=collection_name,
        filter=f'academic_year == "{academic_year}"',
        output_fields=["source_file"],
    )
    return len(results)


# ---------------------------------------------------------------------------
# 删除（回滚）
# ---------------------------------------------------------------------------
def delete_by_academic_year(
    academic_year: str,
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> int:
    """删除指定学年的全部 chunk（回滚操作）。

    参数：
        academic_year: 学年标签，如 "2025-2026"
        collection_name: Milvus collection 名称
        force: 是否跳过确认提示

    返回：删除的 chunk 数量
    """
    client = _get_milvus_client()

    # 先查询影响范围
    results = client.query(
        collection_name=collection_name,
        filter=f'academic_year == "{academic_year}"',
        output_fields=["source_file"],
    )
    count = len(results)

    if count == 0:
        print(f"学年 {academic_year} 无数据，无需删除")
        return 0

    # 收集涉及的文件名（用于同步更新 chunks.json）
    source_files = set(r.get("source_file", "") for r in results)
    print(f"学年 {academic_year} 共 {count} 条 chunk，涉及 {len(source_files)} 个文件")

    if not force:
        confirm = input(f"确认删除学年 {academic_year} 的全部 {count} 条 chunk？(y/N): ")
        if confirm.lower() != "y":
            print("已取消")
            return 0

    # 执行删除
    client.delete(
        collection_name=collection_name,
        filter=f'academic_year == "{academic_year}"',
    )
    print(f"✅ 已删除学年 {academic_year} 的 {count} 条 chunk")

    # 同步更新 chunks.json（移除对应学年+文件的记录）
    _sync_chunks_json_remove(source_files, academic_year)

    return count


# ---------------------------------------------------------------------------
# 增量导入
# ---------------------------------------------------------------------------
def import_files(
    dir_path: str,
    academic_year: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """增量导入新文件到指定学年。

    参数：
        dir_path: 新文件所在目录
        academic_year: 学年标签，如 "2025-2026"
        collection_name: Milvus collection 名称

    返回：导入的 chunk 数量
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.exists():
        print(f"❌ 目录不存在: {dir_path}")
        return 0

    # 1. 预处理
    pipeline = PreprocessingPipeline()
    chunks = pipeline.process_directory(dir_path)
    if not chunks:
        print("未找到可处理的文件")
        return 0

    # 2. 覆盖 academic_year 标签
    for chunk in chunks:
        chunk.metadata["academic_year"] = academic_year

    # 3. 转为 Document 并追加到 Milvus（drop_old=False）
    docs = []
    chunk_records = []
    core_fields = (
        "source_file", "business_tag", "academic_year", "chapter",
        "chunk_strategy", "heading_path",
        "doc_type", "effective_date", "scope",
    )

    # 获取当前 chunks.json 中的最大序号（用于生成唯一 chunk_id）
    existing_chunks = _load_chunks_json()
    base_idx = len(existing_chunks)

    for idx, chunk in enumerate(chunks):
        if not chunk.content.strip():
            continue

        metadata = {}
        for k in core_fields:
            v = chunk.metadata.get(k)
            if v is None:
                metadata[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                metadata[k] = str(v)
            else:
                metadata[k] = ""

        docs.append(Document(page_content=chunk.content, metadata=metadata))
        chunk_records.append({
            "chunk_id": f"{chunk.source_file}__chunk_{base_idx + idx}",
            "text": chunk.content,
            "source_file": chunk.source_file,
            "metadata": metadata,
        })

    if not docs:
        print("所有文件内容为空，无数据可导入")
        return 0

    # 追加到 Milvus
    embeddings = build_embeddings()
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": MILVUS_URI},
    )
    vectorstore.add_documents(docs)
    print(f"✅ 已追加 {len(docs)} 条 chunk 到 Milvus（学年: {academic_year}）")

    # 追加到 chunks.json
    _sync_chunks_json_add(chunk_records)

    return len(docs)


# ---------------------------------------------------------------------------
# 重建（先删后导）
# ---------------------------------------------------------------------------
def rebuild_academic_year(
    academic_year: str,
    dir_path: str,
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> int:
    """重建指定学年：先删除旧数据，再从目录重新导入。

    参数：
        academic_year: 学年标签
        dir_path: 包含该学年所有文件的目录
        collection_name: Milvus collection 名称
        force: 是否跳过确认提示

    返回：重建后的 chunk 数量
    """
    print(f"=== 重建学年 {academic_year} ===")
    print(f"源目录: {dir_path}")

    # 1. 先查询是否存在旧数据
    existing_count = count_by_academic_year(academic_year, collection_name)

    # 2. 删除旧数据
    deleted = delete_by_academic_year(academic_year, collection_name, force=force)

    # 3. 如果有旧数据但删除返回 0，说明用户取消了
    if existing_count > 0 and deleted == 0:
        print("❌ 删除已取消，重建中止")
        return 0

    print(f"已删除 {deleted} 条旧 chunk")

    # 4. 重新导入
    imported = import_files(dir_path, academic_year, collection_name)
    print(f"已导入 {imported} 条新 chunk")

    return imported


# ---------------------------------------------------------------------------
# chunks.json 同步辅助
# ---------------------------------------------------------------------------
def _load_chunks_json() -> list[dict]:
    """加载 data/processed/chunks.json。"""
    path = Path("data/processed/chunks.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunks_json(records: list[dict]):
    """保存 data/processed/chunks.json。"""
    path = Path("data/processed/chunks.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _sync_chunks_json_remove(source_files: set[str], academic_year: str = ""):
    """从 chunks.json 中移除指定学年 + 指定文件的记录。

    参数：
        source_files: 要移除的文件名集合
        academic_year: 学年标签（空字符串表示不按学年过滤）
    """
    records = _load_chunks_json()
    original_count = len(records)
    if academic_year:
        # 同时匹配 source_file 和 academic_year，避免误删其他学年的记录
        records = [
            r for r in records
            if not (
                r.get("source_file", "") in source_files
                and r.get("metadata", {}).get("academic_year", "") == academic_year
            )
        ]
    else:
        records = [r for r in records if r.get("source_file", "") not in source_files]
    removed = original_count - len(records)
    if removed > 0:
        _save_chunks_json(records)
        print(f"✅ chunks.json 已同步移除 {removed} 条记录")


def _sync_chunks_json_add(new_records: list[dict]):
    """向 chunks.json 追加新记录。"""
    records = _load_chunks_json()
    records.extend(new_records)
    _save_chunks_json(records)
    print(f"✅ chunks.json 已追加 {len(new_records)} 条记录（总计 {len(records)}）")
