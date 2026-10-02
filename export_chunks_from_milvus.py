"""从已有 Milvus 向量库导出 chunk 语料到 data/processed/chunks.json。

用途：
- 当 Milvus 已存在但 chunks.json 缺失时，无需重跑预处理 + Embedding，
  直接从向量库反查 text + metadata 导出。
- 导出结果与 build_vector_store() 的副作用完全一致（chunk_id / text / source_file / metadata）。

使用方式：
    python export_chunks_from_milvus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vectorstore import load_vector_store, COLLECTION_NAME, MILVUS_URI


def export_chunks(output_path: str = "data/processed/chunks.json"):
    """从 Milvus 反查所有 chunk，导出到 json。"""
    print(f"📦 连接 Milvus（URI: {MILVUS_URI}）...")
    vectorstore = load_vector_store()

    # 获取 Milvus 客户端和 collection 名
    client = vectorstore._milvus_client
    col_name = vectorstore.collection_name

    # 查询所有记录（取 text + metadata）
    output_fields = ["text", "source_file", "business_tag", "academic_year",
                     "chapter", "chunk_strategy", "heading_path"]
    results = client.query(
        collection_name=col_name,
        filter="",  # 无条件 = 全量
        output_fields=output_fields,
        limit=10000,  # 安全上限
    )

    print(f"📊 从 Milvus 查询到 {len(results)} 条记录")

    # 组装 chunk_records
    chunk_records = []
    for i, r in enumerate(results):
        source_file = r.get("source_file", "")
        chunk_records.append({
            "chunk_id": f"{source_file}__chunk_{i}",
            "text": r.get("text", ""),
            "source_file": source_file,
            "metadata": {
                "source_file": source_file,
                "business_tag": r.get("business_tag", ""),
                "academic_year": r.get("academic_year", ""),
                "chapter": r.get("chapter", ""),
                "chunk_strategy": r.get("chunk_strategy", ""),
                "heading_path": r.get("heading_path", ""),
            },
        })

    # 落盘
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)

    print(f"✅ chunk 语料已导出: {output_path}（{len(chunk_records)} 条）")
    return len(chunk_records)


if __name__ == "__main__":
    export_chunks()
