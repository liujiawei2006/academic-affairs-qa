"""融合策略 + HybridRetriever：把多路召回结果合并为统一排序。

两种融合方式：
1. RRF（Reciprocal Rank Fusion）：基于排名，鲁棒、几乎不用调参，首选
2. 加权归一化分数融合：基于分数，需要调权重配比，备选

HybridRetriever 对外暴露 .invoke(query) → list[Document]，评估代码零改动。
"""
from langchain_core.documents import Document


def rrf_fuse(
    doc_lists: list[list[Document]],
    k: int = 60,
    top_k: int = 10,
    weights: list[float] | None = None,
) -> list[Document]:
    """RRF（Reciprocal Rank Fusion）倒数排名融合。

    原理：score(d) = Σ weight_i / (k + rank_i(d))
    - 完全抛弃分数，只看排名，规避量纲差异
    - k 控制头部抑制力度：k 越小越强调头部，k 越大越民主
    - 论文推荐 k=60 起步
    - weights 控制每路权重（默认等权）

    参数：
        doc_lists: 多路召回结果列表，每路按分数从高到低排序
        k: 平滑常数，默认 60
        top_k: 返回前多少条
        weights: 每路权重列表，默认等权
    """
    if weights is None:
        weights = [1.0] * len(doc_lists)
    if len(weights) != len(doc_lists):
        raise ValueError(
            f"权重数量({len(weights)})与检索路数({len(doc_lists)})不匹配"
        )

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list):
            # 用 page_content 做去重键（同一 chunk 的 text 完全相同）
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            scores[key] += weight / (k + rank + 1)  # rank 从 0 开始，+1 转为 1-based

    # 按 RRF 分数降序排列
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        doc = doc_map[key]
        # 融合后 metadata 带 rrf_score，便于后续分析
        merged_meta = dict(doc.metadata)
        merged_meta["rrf_score"] = scores[key]
        results.append(
            Document(page_content=doc.page_content, metadata=merged_meta)
        )
    return results


def weighted_fuse(
    doc_lists: list[list[Document]],
    weights: list[float] | None = None,
    top_k: int = 10,
) -> list[Document]:
    """加权归一化分数融合。

    原理：
    1. 每路内部 min-max 归一化到 [0, 1]（消除量纲差异）
    2. 按权重加权求和

    参数：
        doc_lists: 多路召回结果，每路需带 metadata["score"]
        weights: 各路权重，默认等权
        top_k: 返回前多少条
    """
    if weights is None:
        weights = [1.0 / len(doc_lists)] * len(doc_lists)
    if len(weights) != len(doc_lists):
        raise ValueError(
            f"权重数量({len(weights)})与检索路数({len(doc_lists)})不匹配"
        )

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list, weight in zip(doc_lists, weights):
        if not doc_list:
            continue

        # 提取分数
        raw_scores = [d.metadata.get("score", 0.0) for d in doc_list]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max_s - min_s

        for doc, raw_s in zip(doc_list, raw_scores):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0

            # min-max 归一化后加权
            norm_s = (raw_s - min_s) / score_range if score_range > 0 else 1.0
            scores[key] += weight * norm_s

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        doc = doc_map[key]
        merged_meta = dict(doc.metadata)
        merged_meta["fused_score"] = scores[key]
        results.append(
            Document(page_content=doc.page_content, metadata=merged_meta)
        )
    return results


class HybridRetriever:
    """混合检索器：向量路 + BM25 路 → 融合 → top-K。

    用法：
        from src.retrieval import HybridRetriever, BM25Retriever
        from src.vectorstore import load_vector_store

        vec_store = load_vector_store()
        vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever()

        # RRF 融合
        hybrid = HybridRetriever(vec_retriever, bm25_retriever, fusion="rrf", k=60)
        docs = hybrid.invoke("计算机专业毕业需要多少学分")

        # 加权融合
        hybrid = HybridRetriever(vec_retriever, bm25_retriever,
                                 fusion="weighted", weights=[0.5, 0.5])
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        fusion: str = "rrf",
        k: int = 60,
        weights: list[float] | None = None,
        top_k: int = 10,
    ):
        """
        参数：
            vector_retriever: 向量检索器（需支持 .invoke()）
            bm25_retriever: BM25 检索器（需支持 .invoke()）
            fusion: 融合策略，"rrf" 或 "weighted"
            k: RRF 平滑常数（仅 fusion="rrf" 时生效）
            weights: 权重配比 [向量权重, BM25权重]（仅 fusion="weighted" 时生效）
            top_k: 返回前多少条
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion = fusion
        self.k = k
        self.weights = weights
        self.top_k = top_k

    def invoke(self, query: str) -> list[Document]:
        """融合两路检索结果。"""
        # 两路各自召回
        vec_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 融合
        if self.fusion == "rrf":
            return rrf_fuse(
                [vec_docs, bm25_docs],
                k=self.k,
                top_k=self.top_k,
            )
        elif self.fusion == "weighted":
            return weighted_fuse(
                [vec_docs, bm25_docs],
                weights=self.weights,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"未知融合策略: {self.fusion}，可选 rrf / weighted")
