"""Rerank 重排模块：SiliconFlow Rerank API 封装 + RerankRetriever 链路包装器。

设计要点：
- SiliconFlowReranker：调用 SiliconFlow 的 BAAI/bge-reranker-v2-m3 API，
  输入 (query, documents) 列表，输出按相关性降序排列的结果
- RerankRetriever：包装任意 base_retriever，先取 top-N 候选，再调 reranker 精排取 top-K
- deduplicate_by_source：文档级去重，同一 source_file 最多保留 max_per_source 个 chunk，
  解决"同一文档占满 Top-5 槽位"的召回浪费问题
- topic_aware_boost：主题感知的时效补偿（联合条件），仅当
  「query 含强时效信号」+「doc_type=通知」+「business_tag 与 query 主题一致」三者同时满足时，
  才对该文档做 tie-break 量级（默认 0.08）的分数提升，避免无差别提升"通知"类导致的跨主题误伤。
  注意：boost 只能重排已在候选池内的文档，无法召回 hybrid 阶段就没进池的文档。
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
            base_retriever=hybrid_retriever,
            reranker=reranker,
            fetch_k=10,   # 粗排取多少候选（推荐 10，与 base retriever 输出对齐）
            top_k=5,      # 精排后保留多少
            max_per_source=2,  # 同一 source_file 最多保留 2 个 chunk（文档级去重）
        )
        docs = rerank_retriever.invoke("计算机专业毕业需要多少学分")
    """

    def __init__(
        self,
        base_retriever,
        reranker: SiliconFlowReranker,
        fetch_k: int = 10,
        top_k: int = 5,
        max_per_source: int | None = None,
        enable_boost: bool = False,
        boost: float = 0.08,
        current_academic_year: str = "2025-2026",
    ):
        """
        参数：
            base_retriever: 基础检索器（如 HybridRetriever），
                需支持 .invoke() 且返回带 source_file metadata 的 Document
            reranker: Reranker 实例
            fetch_k: 粗排阶段取多少候选（入口宽度），推荐 10（与 base retriever 输出对齐，
                扩大至 20 收益极小且增加 API 延迟）
            top_k: 精排后保留条数（出口宽度），推荐 5
            max_per_source: 同一 source_file 最多保留几个 chunk（文档级去重）。
                None=不去重（默认），推荐设为 2。
                去重后若结果不足 top_k，会从剩余候选中按 rerank 分数补充。
            enable_boost: 是否启用主题感知时效补偿（默认关闭，需对照实验验证净收益）
            boost: 时效补偿量级（tie-break 级，默认 0.08，远小于 rerank 分差，不主导排序）
            current_academic_year: 当前学年，用于 boost 时的额外微调匹配
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.max_per_source = max_per_source
        self.enable_boost = enable_boost
        self.boost = boost
        self.current_academic_year = current_academic_year

    def invoke(self, query: str) -> list[Document]:
        """粗排 → 精排 → （可选）文档级去重 → top-K。"""
        # 1. 粗排：从 base retriever 取候选
        #    去重开启时扩大候选池，确保去重后仍有 top_k 条不同文档
        base_fetch = self.fetch_k
        if self.max_per_source is not None:
            # 至少取 top_k * 4 条候选，以抵消同文档多 chunk 导致的槽位浪费
            base_fetch = max(self.fetch_k, self.top_k * 4)
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # 去重开启时不截断，充分利用 base retriever 扩展后的全部候选
        if self.max_per_source is None:
            candidates = candidates[: base_fetch]

        # 2. 精排：reranker 按 (query, doc) 相关性重排
        #    去重开启时，让 reranker 返回全部候选（不截断），后续由去重步骤控制输出量
        rerank_n = len(candidates) if self.max_per_source is not None else self.top_k
        reranked = self.reranker.rerank(query, candidates, top_n=rerank_n)

        # 3. 业务规则：主题感知时效补偿（联合条件 boost）
        #    仅当「query 含强时效信号」+「doc_type=通知」+「business_tag 与 query 主题一致」
        #    三者同时满足时，才做 tie-break 量级提升，避免无差别提升"通知"导致跨主题退步。
        #    默认关闭，需 --boost 对照实验验证净收益为正后再启用。
        if self.enable_boost:
            reranked = topic_aware_boost(
                reranked,
                query,
                current_academic_year=self.current_academic_year,
                boost=self.boost,
            )

        # 4. 文档级去重（可选）
        if self.max_per_source is not None:
            reranked = deduplicate_by_source(
                reranked,
                max_per_source=self.max_per_source,
                top_k=self.top_k,
            )
        else:
            reranked = reranked[: self.top_k]

        return reranked


def infer_query_business_tag(query: str) -> str:
    """从 query 推断业务主题标签，复用 MetadataExtractor 的关键词映射体系。

    保证 query 侧与文档侧用同一套 business_tag 词表，使 boost 的
    "主题一致"判定可靠。按 BUSINESS_TAG_PRIORITY 顺序匹配（越靠前越具体），
    命中即返回；无法判断时返回空字符串（此时不做任何 boost）。
    """
    from src.preprocessing.metadata import MetadataExtractor

    for tag in MetadataExtractor.BUSINESS_TAG_PRIORITY:
        keywords = MetadataExtractor.BUSINESS_TAG_KEYWORDS.get(tag, [])
        if any(kw in query for kw in keywords):
            return tag
    return ""


def topic_aware_boost(
    docs: list[Document],
    query: str,
    current_academic_year: str = "2025-2026",
    boost: float = 0.08,
) -> list[Document]:
    """主题感知的时效补偿（联合条件 boost）。

    相比旧版 current_semester_boost / metadata_boost 的三点修正：
    1. 联合条件：必须 doc_type=通知 AND business_tag == query 主题，
       只提升"和问题同主题的通知"，消除跨主题误伤（旧版无差别提升所有通知）。
    2. tie-break 量级：boost 默认 0.08，远小于 rerank 分差，只在分数接近时微调，
       不主导排序（旧版 boost=0.5 直接覆盖精排结果）。
    3. 收窄触发信号：只在 query 含强时效词时触发，删除"申请"等宽泛词
       （旧版几乎所有教务问题都含"申请"，触发面过宽）。

    局限：boost 只能重排已在候选池内的文档，无法召回 hybrid 阶段就没进池的文档
    （如 Q19/Q58 的目标文档只有 4 个短 chunk，可能压根没进 top-N 候选）。

    参数：
        docs: 按 rerank 分数降序排列的 Document 列表
        query: 用户查询（用于推断主题 + 检测时效信号）
        current_academic_year: 当前学年，命中时额外微调 +0.02
        boost: 基础补偿量（tie-break 级，默认 0.08）

    返回：
        重新排序后的 Document 列表（metadata 追加 metadata_boost 记录补偿量）
    """
    if not docs:
        return docs

    # 1. 推断 query 主题；无法判断主题时不加权（避免误伤）
    query_tag = infer_query_business_tag(query)
    if not query_tag:
        return docs

    # 2. 收窄的强时效信号（删除"申请"等宽泛词）
    time_signals = [
        "什么时候", "何时", "截止", "本学期", "这学期", "这学年",
        "几号", "多久", "时间范围", "报名", "日期",
    ]
    if not any(sig in query for sig in time_signals):
        return docs

    boosted = []
    for doc in docs:
        meta = dict(doc.metadata)
        score = float(meta.get("rerank_score", 0.0))
        doc_type = meta.get("doc_type", "")
        business_tag = meta.get("business_tag", "")
        effective_date = meta.get("effective_date", "")
        actual_boost = 0.0

        # 联合条件：通知类 + 与 query 同主题
        if doc_type == "通知" and business_tag == query_tag:
            actual_boost = boost
            # 当前学年额外微调
            if current_academic_year and current_academic_year in effective_date:
                actual_boost += 0.02

        meta["rerank_score"] = score + actual_boost
        meta["metadata_boost"] = actual_boost
        boosted.append(Document(page_content=doc.page_content, metadata=meta))

    # 按补偿后分数降序重排
    boosted.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return boosted


def metadata_boost(
    docs: list[Document],
    query: str,
    time_boost: float = 0.3,
    scope_boost: float = 0.2,
) -> list[Document]:
    """时效/范围感知的分数补偿。

    当查询包含时效性信号（"什么时候""时间""本学期""申请"等）时，
    提升 doc_type="通知" 且 effective_date 匹配的文档分数，
    解决"长效制度压制本学期通知"的问题。

    当查询包含范围信号（"计算机学院""屏峰"等）时，
    提升 scope 匹配的文档分数。

    参数：
        docs: 按 rerank 分数降序排列的 Document 列表
        query: 用户查询
        time_boost: 时效匹配时的分数补偿量（默认 0.3）
        scope_boost: 范围匹配时的分数补偿量（默认 0.2）

    返回：
        重新排序后的 Document 列表
    """
    if not docs:
        return docs

    # 检测查询中的时效信号
    time_signals = ["什么时候", "时间", "学期", "本学期", "这学年", "今年", "申请", "报名", "截止", "日期"]
    has_time_signal = any(sig in query for sig in time_signals)

    # 检测查询中的范围信号
    scope_signals = {
        "学院": ["计算机学院", "软件学院", "计算机"],
        "校区": ["屏峰", "朝晖", "莫干山"],
    }
    query_scope = ""
    for scope_val, keywords in scope_signals.items():
        if any(kw in query for kw in keywords):
            query_scope = scope_val
            break

    # 无信号时直接返回（避免不必要的计算）
    if not has_time_signal and not query_scope:
        return docs

    # 提取查询中的学年信息（用于精确匹配）
    import re
    query_year_match = re.search(r"(\d{4})[-—](\d{4})", query)
    query_year = f"{query_year_match.group(1)}-{query_year_match.group(2)}" if query_year_match else ""

    boosted = []
    for doc in docs:
        meta = doc.metadata
        score = float(meta.get("rerank_score", 0.0))
        doc_type = meta.get("doc_type", "")
        effective_date = meta.get("effective_date", "")
        doc_scope = meta.get("scope", "")
        boost = 0.0

        # 时效补偿：查询含时效信号 + 文档是通知类
        if has_time_signal and doc_type == "通知":
            boost += time_boost
            # 学年精确匹配额外加分
            if query_year and query_year in effective_date:
                boost += 0.1

        # 范围补偿：查询含范围信号 + 文档 scope 匹配
        if query_scope and doc_scope == query_scope:
            boost += scope_boost

        meta = dict(meta)
        meta["rerank_score"] = score + boost
        meta["metadata_boost"] = boost
        boosted.append(Document(page_content=doc.page_content, metadata=meta))

    # 按补偿后分数降序重排
    boosted.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return boosted


def deduplicate_by_source(
    docs: list[Document],
    max_per_source: int = 2,
    top_k: int | None = None,
) -> list[Document]:
    """文档级去重：同一 source_file 最多保留 max_per_source 个 chunk。

    按输入顺序（即 rerank 分数降序）遍历，每个 source_file 累计不超过
    max_per_source 个 chunk。去重后若不足 top_k，不再补充（因为输入已经是
    扩大后的候选集，后续 chunk 分数更低，补充价值有限）。

    参数：
        docs: 按 rerank 分数降序排列的 Document 列表
        max_per_source: 同一 source_file 最多保留几个 chunk
        top_k: 去重后最多保留几条（None=不限制）

    返回：
        去重后的 Document 列表，保持原始排序顺序
    """
    source_counts: dict[str, int] = {}
    deduped = []

    for doc in docs:
        source = doc.metadata.get("source_file", "")
        count = source_counts.get(source, 0)
        if count >= max_per_source:
            continue  # 该文档已达上限，跳过
        source_counts[source] = count + 1
        deduped.append(doc)
        if top_k is not None and len(deduped) >= top_k:
            break

    return deduped
