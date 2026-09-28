"""Milvus 向量存储：支持业务标签过滤。"""
import os
import re
import unicodedata
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
import requests

from src.preprocessing import DocumentChunk


# Embedding 配置
EMBEDDING_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
EMBEDDING_API_KEY = os.getenv("LLM_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_academic.db")
COLLECTION_NAME = "academic_affairs"


def sanitize_text(text: str, max_length: int = 2000) -> str:
    """温和清洗文本，保留语义完整性。

    清洗步骤：
    1. UTF-8 非法字节忽略（编码时 errors='ignore'）
    2. NFKC 规范化（统一全角/半角、兼容字符）
    3. 移除控制符、格式符、零宽、BOM、私用区、代理项
    4. 合并异常空白
    5. 检查空文本和超长
    """
    # 1. UTF-8 非法字节忽略
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # 2. NFKC 规范化
    text = unicodedata.normalize("NFKC", text)

    # 3. 移除特殊 Unicode 字符
    def _keep_char(c: str) -> bool:
        cp = ord(c)
        cat = unicodedata.category(c)
        # 移除控制符 (Cc)、格式符 (Cf)、未分配 (Cn)、私用区 (Co)、代理项 (Cs)
        if cat.startswith("C"):
            # 保留常见空白：换行、回车、制表符
            if c in "\n\r\t":
                return True
            return False
        # 移除零宽字符 (U+200B-U+200F, U+FEFF 等)
        if 0x200B <= cp <= 0x200F or cp == 0xFEFF:
            return False
        # 移除私用区 (U+E000-U+F8FF)
        if 0xE000 <= cp <= 0xF8FF:
            return False
        return True

    text = "".join(c for c in text if _keep_char(c))

    # 4. 合并异常空白（保留单个换行，合并多个空格）
    text = re.sub(r"[^\S\n\r]+", " ", text)  # 合并空格（保留换行）
    text = re.sub(r"\n{3,}", "\n\n", text)  # 合并多个换行

    # 5. 检查空文本和超长
    text = text.strip()
    if not text:
        return ""
    if len(text) > max_length:
        text = text[:max_length]

    return text


class SiliconFlowEmbeddings(Embeddings):
    """兼容 SiliconFlow 的 Embedding 封装，带失败兜底链路。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._failed_texts = []  # 记录失败的文本，供后续扩展备用 API

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """单次调用 Embedding API。"""
        resp = self._session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in results]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批调用 Embedding API，带失败兜底链路。

        兜底流程：
        1. 原文本调用
        2. 失败 → 温和清洗后重试
        3. 仍失败 → 分段重试（拆成更小片段）
        4. 仍失败 → 放弃此文本，记录日志
        """
        batch_size = 16
        max_text_length = 2000
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_idx = i // batch_size + 1

            # 截断过长文本
            batch = [t[:max_text_length] if len(t) > max_text_length else t for t in batch]

            # 尝试 1: 原文本
            try:
                embeddings = self._call_api(batch)
                all_embeddings.extend(embeddings)
                continue
            except Exception as e1:
                print(f"⚠️ 批次 {batch_idx} 原文本失败: {e1}")

            # 尝试 2: 温和清洗后重试
            cleaned_batch = [sanitize_text(t, max_text_length) for t in batch]
            # 过滤空文本
            valid_indices = [j for j, t in enumerate(cleaned_batch) if t]
            if not valid_indices:
                print(f"⚠️ 批次 {batch_idx} 清洗后全部为空，放弃")
                # 用零向量占位（或跳过）
                all_embeddings.extend([[0.0] * 1024] * len(batch))
                continue

            cleaned_batch = [cleaned_batch[j] for j in valid_indices]
            try:
                embeddings = self._call_api(cleaned_batch)
                # 还原顺序（无效位置用零向量占位）
                result = []
                emb_idx = 0
                for j in range(len(batch)):
                    if j in valid_indices:
                        result.append(embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        result.append([0.0] * 1024)  # 零向量占位
                all_embeddings.extend(result)
                print(f"✅ 批次 {batch_idx} 温和清洗后成功")
                continue
            except Exception as e2:
                print(f"⚠️ 批次 {batch_idx} 温和清洗后仍失败: {e2}")

            # 尝试 3: 分段重试（把每个文本拆成更小的片段）
            try:
                sub_embeddings = []
                for j, text in enumerate(cleaned_batch):
                    # 按 500 字符分段
                    sub_texts = [text[k:k+500] for k in range(0, len(text), 500)]
                    sub_texts = [t for t in sub_texts if t.strip()]
                    if not sub_texts:
                        sub_embeddings.append([0.0] * 1024)
                        continue
                    sub_embs = self._call_api(sub_texts)
                    # 取平均作为该文本的向量
                    avg_emb = [sum(e[d] for e in sub_embs) / len(sub_embs) for d in range(1024)]
                    sub_embeddings.append(avg_emb)

                # 还原顺序
                result = []
                emb_idx = 0
                for j in range(len(batch)):
                    if j in valid_indices:
                        result.append(sub_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        result.append([0.0] * 1024)
                all_embeddings.extend(result)
                print(f"✅ 批次 {batch_idx} 分段重试后成功")
                continue
            except Exception as e3:
                print(f"❌ 批次 {batch_idx} 分段重试仍失败: {e3}，放弃此批次")
                self._failed_texts.extend(batch)
                # 零向量占位
                all_embeddings.extend([[0.0] * 1024] * len(batch))

        if self._failed_texts:
            print(f"\n⚠️ 共 {len(self._failed_texts)} 条文本 Embedding 失败，已用零向量占位")
            print("   可扩展：接入备用 API 或本地模型重新处理这些文本")

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings() -> SiliconFlowEmbeddings:
    """构建 Embedding 实例。"""
    return SiliconFlowEmbeddings(EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL)


def build_vector_store(chunks: list[DocumentChunk], drop_old: bool = True) -> Milvus:
    """将文档片段存入 Milvus，支持业务标签字段。"""
    import json
    from langchain_core.documents import Document

    docs = []
    for chunk in chunks:
        # 过滤空内容
        if not chunk.content.strip():
            continue

        # 处理元数据：只保留对检索有用的核心字段（Milvus 要求字段一致）
        core_fields = ("source_file", "business_tag", "academic_year", "chapter", "chunk_strategy", "heading_path")
        metadata = {}
        for k in core_fields:
            v = chunk.metadata.get(k)
            if v is None:
                metadata[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                metadata[k] = str(v)
            else:
                metadata[k] = ""

        doc = Document(
            page_content=chunk.content,
            metadata=metadata,
        )
        docs.append(doc)

    # 创建 Milvus 向量库
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=build_embeddings(),
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=drop_old,
    )

    print(f"✅ 向量已存入 Milvus（URI: {MILVUS_URI}），共 {len(docs)} 条记录")
    return vectorstore


def load_vector_store() -> Milvus:
    """加载已存在的 Milvus 向量库。"""
    embeddings = build_embeddings()
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
    )
    return vectorstore


def search_with_filter(
    query: str,
    business_tag: Optional[str] = None,
    top_k: int = 5,
) -> list:
    """带业务标签过滤的相似度检索。"""
    vectorstore = load_vector_store()

    search_kwargs = {"k": top_k}
    if business_tag:
        # Milvus 支持通过 metadata 过滤
        search_kwargs["filter"] = f'business_tag == "{business_tag}"'

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)
