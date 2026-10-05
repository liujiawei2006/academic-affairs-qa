"""Query 改写模块：LLM 改写器 + RewriteRetriever。

两个组件：
1. CachedLLMRewriter：LLM 多查询生成（口语术语理解由 prompt 承担），带本地 JSON 缓存
2. RewriteRetriever：多查询 → 每变体走融合召回 → 分数加权二次融合，兼容 .invoke()
"""
import hashlib
import json
import os
import re
from pathlib import Path

from langchain_core.documents import Document

from src.retrieval.fusion import weighted_fuse


# ---------------------------------------------------------------------------
# 2.1 LLM 改写器 + 多查询生成
# ---------------------------------------------------------------------------
REWRITE_PROMPT = """\
你是高校教务系统的查询改写助手。任务：把学生提问改写为更适合检索的查询。

规则（严格遵守）：
1. 保留所有专有名词和具体名称（如"省政府奖学金特别优秀""计算机科学与技术""浙江工业大学"），绝对不能泛化或省略
2. 只替换口语化表达为正式术语（如"挂科"→"不及格课程"、"保研"→"推荐免试研究生"）
3. 每个变体必须是完整的疑问句，不能是陈述句或短语
4. 只生成 2 个变体，不要更多
5. 每个变体占一行，不要编号，不要解释，不要输出其他内容

原始问题：{query}

改写结果："""


class CachedLLMRewriter:
    """LLM 改写器，带本地 JSON 缓存。

    缓存 key = query 文本的 MD5 hash，避免重复调 API。
    每次写入都立即落盘，评估中断也能恢复。
    """

    def __init__(
        self,
        cache_path: str = "data/processed/rewrite_cache.json",
        temperature: float = 0.1,
    ):
        self.cache_path = Path(cache_path)
        self.cache: dict[str, list[str]] = {}

        # 加载已有缓存
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
            print(f"  已加载改写缓存: {len(self.cache)} 条")

        # 初始化 LLM（复用 .env 配置）
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=temperature,
            request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )

    @staticmethod
    def _make_key(query: str) -> str:
        """query 文本做 MD5 key，避免特殊字符问题。"""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def rewrite(self, query: str) -> list[str]:
        """获取改写结果（优先走缓存）。

        返回 2~3 个查询变体；若 LLM 失败则返回空列表。
        """
        key = self._make_key(query)
        if key in self.cache:
            return self.cache[key]

        # 调 LLM
        result = self._call_llm(query)

        # 写入缓存并立即落盘
        self.cache[key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

        return result

    def _call_llm(self, query: str) -> list[str]:
        """调用 LLM 生成改写变体。"""
        prompt = REWRITE_PROMPT.format(query=query)
        try:
            response = self.llm.invoke(prompt)
            return _parse_rewrites(response.content)
        except Exception as e:
            print(f"  LLM 改写失败: {e}，返回空列表")
            return []

    def clear_cache(self) -> None:
        """清空缓存（修改 prompt 后需要调用）。"""
        self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
        print(f"  改写缓存已清空")


def _parse_rewrites(llm_output: str) -> list[str]:
    """解析 LLM 输出，每行一个改写查询。

    防御性处理：去掉编号前缀、空行、markdown 格式。
    """
    lines = llm_output.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 去掉编号前缀 "1. " "1、" "1) " 等
        line = re.sub(r"^\d+[\.\、\)\] ]+\s*", "", line)
        # 去掉 markdown 格式
        line = line.strip("*`").strip()
        if line:
            cleaned.append(line)
    return cleaned


# ---------------------------------------------------------------------------
# 2.3 RewriteRetriever：改写接入检索链路
# ---------------------------------------------------------------------------
class RewriteRetriever:
    """改写检索器：多查询改写 → 每变体走融合召回 → 分数加权二次融合。

    对外接口与 HybridRetriever 完全一致：.invoke(query) → list[Document]
    评估代码零改动。

    设计要点（解决双层融合冲突）：
        第一层 base_retriever（如 HybridRetriever）用分数融合输出带分数的 top-k；
        第二层不再用 RRF 排名融合（会丢弃第一层分数、把精排降维成纯排名），
        而是提取每路的融合分数做 min-max 归一化后加权求和，保留分数区分度。
        同时原始 query 权重 > 所有变体权重之和，确保原始 query 排名主导、
        变体只在原始 query 召回不足时做补强，不会稀释正确排名。

    数据流：
        原始 query
        → CachedLLMRewriter（生成 ≤max_variants 个变体，口语术语由 prompt 理解）
        → 每个变体分别走 base_retriever.invoke()
        → 分数加权二次融合（原始 query 权重 > 变体权重之和）→ top-k
    """

    def __init__(
        self,
        base_retriever,
        rewriter: CachedLLMRewriter,
        top_k: int = 10,
        original_weight: float = 3.0,
        variant_weight: float = 1.0,
        max_variants: int = 2,
    ):
        """
        参数：
            base_retriever: 基础检索器（如 HybridRetriever），需支持 .invoke()
            rewriter: LLM 改写器实例
            top_k: 最终返回条数
            original_weight: 原始 query 检索结果的权重（默认 3.0）；
                必须 > 所有变体权重之和（max_variants * variant_weight），
                否则一致的变体噪声会在加权后追平甚至压过原始 query 的正确排名
            variant_weight: 单个 LLM 变体检索结果的权重（默认 1.0）
            max_variants: 最多采用的 LLM 变体数（默认 2），用于约束变体权重之和
        """
        self.base_retriever = base_retriever
        self.rewriter = rewriter
        self.top_k = top_k
        self.original_weight = original_weight
        self.variant_weight = variant_weight
        self.max_variants = max_variants

    @staticmethod
    def _extract_score(doc: Document) -> float:
        """从 base 检索结果里提取第一层融合分数。

        HybridRetriever 按融合策略写不同字段：weighted → fused_score，
        rrf → rrf_score；纯检索器（向量/BM25）可能是 score。
        按优先级回退，都取不到则返回 0.0。
        """
        meta = doc.metadata or {}
        for key in ("fused_score", "rrf_score", "score"):
            if key in meta:
                try:
                    return float(meta[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    def invoke(self, query: str) -> list[Document]:
        """改写 + 多路召回 + 分数加权二次融合。"""
        # 1. LLM 多查询生成（直接吃原始 query，口语术语理解交给 prompt），截断到 max_variants
        llm_variants = self.rewriter.rewrite(query)[: self.max_variants]

        # 2. 构建查询列表：原始 query + LLM 变体（去重，保序）
        all_queries = [query] + llm_variants
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        # 3. 每个变体分别走融合召回，提取分数 + 记录权重
        doc_lists = []
        weights = []
        for q in unique_queries:
            docs = self.base_retriever.invoke(q)
            if not docs:
                continue
            # 把第一层的融合分数回写到 metadata["score"]，
            # 供 weighted_fuse 做 min-max 归一化后加权（保留分数区分度）
            scored_docs = []
            for d in docs:
                meta = dict(d.metadata)
                meta["score"] = self._extract_score(d)
                scored_docs.append(
                    Document(page_content=d.page_content, metadata=meta)
                )
            doc_lists.append(scored_docs)
            # 原始 query 权重更高（去重后变体不会等于原始 query）
            weights.append(
                self.original_weight if q == query else self.variant_weight
            )

        # 4. 分数加权二次融合（不再用 RRF 排名融合，避免丢弃第一层分数）
        if not doc_lists:
            return []
        if len(doc_lists) == 1:
            return doc_lists[0][: self.top_k]

        return weighted_fuse(
            doc_lists,
            weights=weights,
            top_k=self.top_k,
        )
