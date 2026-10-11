"""原文事实校验模块（C3 阶段产出）。

对生成答案做后置校验：
1. 从答案中抽取数值/条件项
2. 回原文 context 做正则匹配
3. 命中不了的标记为"疑似幻觉"

设计原则：
- 不追求完全精确（模型可能改写数值表述），只抓明显编造
- 数值用正则抽取 + 归一化后比对
- 条件项用关键词匹配
- 返回校验报告，不阻断回答输出
"""
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ClaimCheck:
    """单条声明的校验结果。"""

    claim_text: str  # 答案中的声明文本
    claim_type: Literal["value", "condition", "other"]  # 声明类型
    matched: bool  # 是否在原文中找到支撑
    matched_context: str = ""  # 匹配到的原文片段
    confidence: str = "high"  # high / medium / low


@dataclass
class VerificationReport:
    """校验报告。"""

    answer: str
    checks: list[ClaimCheck] = field(default_factory=list)
    hallucination_count: int = 0  # 疑似幻觉条数
    total_claims: int = 0  # 总声明数

    @property
    def hallucination_rate(self) -> float:
        """幻觉率（疑似幻觉条数 / 总声明数）。"""
        if self.total_claims == 0:
            return 0.0
        return self.hallucination_count / self.total_claims

    @property
    def is_clean(self) -> bool:
        """是否有疑似幻觉。"""
        return self.hallucination_count == 0

    def summary(self) -> str:
        """生成可读摘要。"""
        status = "通过" if self.is_clean else f"发现 {self.hallucination_count} 条疑似幻觉"
        return (
            f"校验{status}：共 {self.total_claims} 条声明，"
            f"幻觉率 {self.hallucination_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# 数值抽取与归一化
# ---------------------------------------------------------------------------

# 常见数值模式：绩点、百分比、金额、学分、人数、日期等
_VALUE_PATTERNS = [
    # 绩点：3.8, 3.5 等
    (r"(?:绩点|GPA)[^\d]*(\d+\.?\d*)", "gpa"),
    # 百分比：前10%, 前20% 等
    (r"前\s*(\d+)\s*%", "percent_rank"),
    # 百分比：不超过X%
    (r"不超过\s*(\d+)\s*%", "percent_max"),
    # 金额：2000元, 25元每小时 等
    (r"(\d+)\s*元", "money"),
    # 学分：160学分, 21学分 等
    (r"(\d+)\s*学分", "credit"),
    # 人数：35人, 19人 等
    (r"(\d+)\s*人", "people"),
    # 题数/分值：20个, 2分, 100分 等
    (r"(\d+)\s*(?:个|道|题)", "count"),
    (r"(\d+\.?\d*)\s*分", "score"),
    # 日期：2026年, 8月31日 等
    (r"(\d{4})\s*年", "year"),
    (r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", "date"),
    # 工时：25元每小时
    (r"(\d+)\s*元\s*每\s*小时", "hourly_rate"),
]


def extract_values_from_text(text: str) -> list[tuple[str, str]]:
    """从文本中抽取数值及其类型。

    返回 [(数值字符串, 类型), ...]
    """
    results = []
    for pattern, vtype in _VALUE_PATTERNS:
        for m in re.finditer(pattern, text):
            # 取完整匹配（包含上下文）作为 claim
            full_match = m.group(0)
            results.append((full_match, vtype))
    return results


def normalize_value(value_str: str) -> str:
    """归一化数值字符串，便于比对。

    规则：
    - 去除空格
    - 统一全角→半角
    - 保留核心数字
    """
    # 全角转半角
    value_str = value_str.translate(
        str.maketrans("０１２３４５６７８９．％", "0123456789.%")
    )
    # 去除空白
    value_str = re.sub(r"\s+", "", value_str)
    return value_str


# ---------------------------------------------------------------------------
# 条件项抽取
# ---------------------------------------------------------------------------

# 条件关键词（教务政策高频条件）
_CONDITION_KEYWORDS = [
    "不及格", "挂科", "处分", "违纪", "贫困认定", "有效期",
    "前三位", "第一作者", "自主上报", "带公章", "公示",
    r"前\d+%", r"不低于\s*\d+", r"不超过\s*\d+",
]


def extract_conditions_from_text(text: str) -> list[str]:
    """从文本中抽取条件项。

    返回关键词及其简短上下文（去除引用标注 [N]）。
    """
    found = []
    # 先去除引用标注，避免干扰匹配
    clean_text = re.sub(r"\[\d+\]", "", text)
    for kw in _CONDITION_KEYWORDS:
        for m in re.finditer(kw, clean_text):
            # 提取关键词及其短上下文（前后各 5 字）
            start = max(0, m.start() - 5)
            end = min(len(clean_text), m.end() + 5)
            snippet = clean_text[start:end].strip()
            if snippet and snippet not in found:
                found.append(snippet)
    return found


# ---------------------------------------------------------------------------
# 核心校验逻辑
# ---------------------------------------------------------------------------

def verify(
    answer: str,
    context_docs: list[Document],
) -> VerificationReport:
    """对答案做原文事实校验。

    参数：
        answer: LLM 生成的答案文本
        context_docs: 生成答案时使用的参考资料（Document 列表）

    返回：
        VerificationReport 校验报告
    """
    # 合并所有 context 文本
    all_context_text = "\n".join(doc.page_content for doc in context_docs)
    all_context_norm = normalize_value(all_context_text)

    checks: list[ClaimCheck] = []

    # 1. 数值校验：从答案中抽取数值，回原文查找
    answer_values = extract_values_from_text(answer)
    for value_str, vtype in answer_values:
        value_norm = normalize_value(value_str)
        # 在原文中查找（归一化后匹配）
        if value_norm in all_context_norm:
            # 找到原文片段（用归一化文本截取，避免索引错位）
            idx = all_context_norm.find(value_norm)
            snippet = all_context_norm[max(0, idx - 20):idx + len(value_norm) + 20]
            checks.append(ClaimCheck(
                claim_text=value_str,
                claim_type="value",
                matched=True,
                matched_context=snippet,
            ))
        else:
            # 尝试宽松匹配（只比核心数字）
            core_number = re.sub(r"[^\d.]", "", value_str)
            if core_number and core_number in all_context_text:
                checks.append(ClaimCheck(
                    claim_text=value_str,
                    claim_type="value",
                    matched=True,
                    matched_context=f"（宽松匹配：数字 {core_number} 存在于原文）",
                    confidence="medium",
                ))
            else:
                checks.append(ClaimCheck(
                    claim_text=value_str,
                    claim_type="value",
                    matched=False,
                ))

    # 2. 条件项校验：从答案中抽取条件关键词，回原文查找
    answer_conditions = extract_conditions_from_text(answer)
    # 预计算：去除引用标注后的归一化 context（循环外一次性计算）
    all_context_norm_clean = normalize_value(
        re.sub(r"\[\d+\]", "", all_context_text)
    )
    for cond in answer_conditions:
        cond_norm = normalize_value(cond)
        if cond_norm in all_context_norm_clean:
            checks.append(ClaimCheck(
                claim_text=cond,
                claim_type="condition",
                matched=True,
            ))
        else:
            # 宽松：只要核心关键词在就算通过
            core_kw = re.sub(r"[^\u4e00-\u9fff\w]", "", cond)
            if core_kw and len(core_kw) >= 2 and core_kw in all_context_text:
                checks.append(ClaimCheck(
                    claim_text=cond,
                    claim_type="condition",
                    matched=True,
                    confidence="medium",
                ))
            else:
                checks.append(ClaimCheck(
                    claim_text=cond,
                    claim_type="condition",
                    matched=False,
                ))

    # 3. 统计
    hallucination_count = sum(1 for c in checks if not c.matched)

    return VerificationReport(
        answer=answer,
        checks=checks,
        hallucination_count=hallucination_count,
        total_claims=len(checks),
    )


def format_report(report: VerificationReport) -> str:
    """格式化校验报告为可读文本。"""
    lines = [report.summary(), ""]

    if report.is_clean:
        lines.append("所有声明均在原文中找到支撑。")
        return "\n".join(lines)

    lines.append("疑似幻觉项：")
    for check in report.checks:
        if check.matched:
            continue
        lines.append(f"  - [{check.claim_type}] \"{check.claim_text}\"")
        lines.append(f"    状态：原文中未找到支撑")

    if any(c.matched for c in report.checks):
        lines.append("")
        lines.append("已验证项：")
        for check in report.checks:
            if not check.matched:
                continue
            conf = "" if check.confidence == "high" else f"（{check.confidence}）"
            lines.append(f"  + [{check.claim_type}] \"{check.claim_text}\" {conf}")

    return "\n".join(lines)
