"""Rerank 重排模块：SiliconFlow Rerank API 封装 + RerankRetriever 链路包装器。

设计要点：
- SiliconFlowReranker：调用 SiliconFlow 的 BAAI/bge-reranker-v2-m3 API，
  输入 (query, documents) 列表，输出按相关性降序排列的结果
- RerankRetriever：包装任意 base_retriever，先取 top-N 候选，再调 reranker 精排取 top-K
- 兼容 .invoke() 接口，评估代码零改动
- metadata 透传：rerank 后保留 source_file / chunk_id 等字段，评估匹配不受影响
"""
import os

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

# 加载 .env 文件（确保在读取配置前执行）
load_dotenv()

# SiliconFlow Rerank API 配置（复用 .env 中的 LLM 配置）
RERANK_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
RERANK_API_KEY = os.getenv("LLM_API_KEY", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


class SiliconFlowReranker:
    """SiliconFlow Rerank API 封装。

    用法：
        reranker = SiliconFlowReranker()
        results = reranker.rerank(
            query="计算机专业毕业需要多少学分",
            docs=[doc1, doc2, ...],
            top_n=5,
        )
    """

    def __init__(
        self,
        model: str = RERANK_MODEL,
        base_url: str = RERANK_BASE_URL,
        api_key: str = RERANK_API_KEY,
        timeout: int = 30,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int = 5,
    ) -> list[Document]:
        """对候选文档按 (query, doc) 相关性重排，返回 top_n。

        参数：
            query: 用户查询
            docs: 候选文档列表（来自 base retriever）
            top_n: 重排后保留的条数

        返回：
            按相关性降序排列的 Document 列表，metadata 新增 rerank_score 字段，
            其余 metadata（source_file / chunk_id 等）原样透传。
        """
        if not docs:
            return []

        # 提取文档文本，传给 API
        texts = [d.page_content for d in docs]

        try:
            resp = self._session.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": top_n,
                    "return_documents": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Rerank API 调用失败: {e}，回退到原始排序")
            return docs[:top_n]

        # API 返回 results 按 relevance_score 降序，每项含 index + relevance_score
        results = data.get("results", [])

        reranked = []
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            doc = docs[idx]
            # 透传原始 metadata，追加 rerank_score
            meta = dict(doc.metadata)
            meta["rerank_score"] = float(score)
            reranked.append(
                Document(page_content=doc.page_content, metadata=meta)
            )

        return reranked


class RerankRetriever:
    """Rerank 检索器：base_retriever 粗排取 top-N → reranker 精排取 top-K。

    对外接口与 HybridRetriever / RewriteRetriever 完全一致：.invoke(query) → list[Document]
    评估代码零改动。

    用法：
        from src.retrieval.reranker import SiliconFlowReranker, RerankRetriever

        reranker = SiliconFlowReranker()
        rerank_retriever = RerankRetriever(
            base_retriever=rewrite_retriever,  # 或 hybrid_retriever
            reranker=reranker,
            fetch_k=10,   # 粗排取多少候选（推荐 10，与 base retriever 输出对齐）
            top_k=5,      # 精排后保留多少
        )
        docs = rerank_retriever.invoke("计算机专业毕业需要多少学分")
    """

    def __init__(
        self,
        base_retriever,
        reranker: SiliconFlowReranker,
        fetch_k: int = 10,
        top_k: int = 5,
    ):
        """
        参数：
            base_retriever: 基础检索器（如 RewriteRetriever / HybridRetriever），
                需支持 .invoke() 且返回带 source_file metadata 的 Document
            reranker: Reranker 实例
            fetch_k: 粗排阶段取多少候选（入口宽度），推荐 10（与 base retriever 输出对齐，
                扩大至 20 收益极小且增加 API 延迟）
            top_k: 精排后保留条数（出口宽度），推荐 5
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.top_k = top_k

    def invoke(self, query: str) -> list[Document]:
        """粗排 → 精排 → top-K。"""
        # 1. 粗排：从 base retriever 取更多候选
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # 限制到 fetch_k（base retriever 可能已返回更少）
        candidates = candidates[: self.fetch_k]

        # 2. 精排：reranker 按 (query, doc) 相关性重排
        reranked = self.reranker.rerank(query, candidates, top_n=self.top_k)

        return reranked
