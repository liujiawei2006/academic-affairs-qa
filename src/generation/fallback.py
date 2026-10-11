"""无依据兜底模块（D1 阶段产出）。

定义兜底触发场景与统一话术：
1. 召回为空（检索未返回任何文档）
2. 超范围问题（命中知识库未收录的主题关键词）
3. C3 校验判定无原文支撑（幻觉率过高）

触发兜底时，直接返回统一话术，不调用 LLM 生成。
"""
from dataclasses import dataclass

from langchain_core.documents import Document

from src.generation.verifier import VerificationReport


# ---------------------------------------------------------------------------
# 兜底话术
# ---------------------------------------------------------------------------

FALLBACK_MESSAGE = "未在现行政策中找到依据，建议咨询教务处。"


# ---------------------------------------------------------------------------
# 超范围关键词（知识库仅收录本科生教务相关文档）
# ---------------------------------------------------------------------------

# 这些主题不在当前知识库覆盖范围内，命中时直接兜底
_OUT_OF_SCOPE_KEYWORDS = [
    # 研究生相关
    "研究生", "硕士", "博士", "导师",
    # 培养方案类
    "转专业", "辅修学位", "辅修专业",
    "毕业设计", "毕业论文", "开题", "答辩",
    "补考", "重修", "缓考",
    # 校园生活类
    "宿舍", "住宿", "后勤",
    "社团", "学生会",
    # 其他
    "出国", "留学", "交换生",
]


# ---------------------------------------------------------------------------
# 兜底判定
# ---------------------------------------------------------------------------

@dataclass
class FallbackDecision:
    """兜底判定结果。"""

    should_fallback: bool  # 是否触发兜底
    reason: str = ""  # 触发原因（用于调试/日志）


def check_fallback(
    question: str,
    docs: list[Document],
    verification: VerificationReport | None = None,
) -> FallbackDecision:
    """判断是否需要触发兜底。

    判定顺序（短路逻辑，命中即返回）：
    1. 召回为空 → 兜底
    2. 超范围关键词 → 兜底
    3. C3 校验幻觉率 >= 80% → 兜底

    参数：
        question: 用户问题
        docs: 检索返回的文档列表
        verification: C3 校验报告（生成答案后才有，可为 None）

    返回：
        FallbackDecision(should_fallback, reason)
    """
    # 1. 召回为空
    if not docs:
        return FallbackDecision(True, "召回为空")

    # 2. 超范围问题检测（关键词匹配）
    for kw in _OUT_OF_SCOPE_KEYWORDS:
        if kw in question:
            return FallbackDecision(True, f"超范围问题（命中关键词：{kw}）")

    # 3. C3 校验全部失败（幻觉率 >= 80%）
    if verification is not None and verification.total_claims > 0:
        if verification.hallucination_rate >= 0.8:
            return FallbackDecision(
                True,
                f"校验失败（幻觉率 {verification.hallucination_rate:.0%}）",
            )

    return FallbackDecision(False)


def pre_llm_fallback_check(question: str, docs: list[Document]) -> FallbackDecision:
    """LLM 调用前的前置兜底检查（召回为空 + 超范围）。

    用于在调用 LLM 之前尽早拦截，节省 API 开销。
    不包含 verification 检查（因为 verification 需要 LLM 生成后才能做）。
    """
    # 1. 召回为空
    if not docs:
        return FallbackDecision(True, "召回为空")

    # 2. 超范围问题检测
    for kw in _OUT_OF_SCOPE_KEYWORDS:
        if kw in question:
            return FallbackDecision(True, f"超范围问题（命中关键词：{kw}）")

    return FallbackDecision(False)
