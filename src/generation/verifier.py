"""原文事实校验模块（C3 阶段产出）。

对生成答案做后置校验：
1. 从答案中抽取数值/条件项
2. 回原文 context 做正则匹配
3. 命中不了的标记为"疑似幻觉"

设计原则：
- 不追求完全精确（模型可能改写数值表述），只抓明显编造
- 数值用正则抽取 + 归一化后比对
- 条件项匹配连续短语及其肯否语境
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
    (r"(\d+(?:\.\d+)?)\s*元(?!\s*每\s*小时)", "money"),
    # 学分：160学分, 21学分 等
    (r"(\d+(?:\.\d+)?)\s*学分", "credit"),
    # 人数：35人, 19人 等
    (r"(\d+)\s*人", "people"),
    # 题数/分值：20个, 2分, 100分 等
    (r"(\d+)\s*(?:个|道|题)", "count"),
    (r"(\d+\.?\d*)\s*分", "score"),
    # 日期：2026年, 8月31日 等
    (r"(?<!\d)(\d{4})\s*年", "year"),
    (r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", "date"),
    # 工时：25元每小时
    (r"(\d+(?:\.\d+)?)\s*元\s*每\s*小时", "hourly_rate"),
]


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


_CREDIT_MAJORS = re.compile(r"计算机科学与技术|软件工程|网络工程|数字媒体技术")
_CREDIT_FIELDS = re.compile(
    r"第二课堂|专业必修|专业选修|学科基础|通识|实践教学|辅修|毕业|总学分"
)
_SENTENCE_BREAK = re.compile(r"[。；;！？!?\n]")
# “两专业毕业总学分相同，均为 166 学分”中的后半句沿用前面的范围。
_CLAUSE_BREAK = re.compile(r"[，,](?!\s*(?:均为|同为|都是|均需|各为))")
_CITATION = re.compile(r"\[(\d+)\]")
_NEGATION = re.compile(r"(?:并非|不是|不需要|不必|无需|无须|不用|不得|不能|没有|未曾|未|不|无)$")


def _credit_scope(text: str) -> tuple[set[str], str]:
    fields = _CREDIT_FIELDS.findall(text)
    field = fields[-1] if fields else ""
    return set(_CREDIT_MAJORS.findall(text)), "毕业" if field == "总学分" else field


def _scoped_clauses(sentence: str):
    previous_majors = set()
    for clause in _CLAUSE_BREAK.split(_CITATION.sub("", sentence)):
        yield normalize_value(clause), previous_majors
        named_majors = set(_CREDIT_MAJORS.findall(clause))
        if named_majors:
            previous_majors = named_majors if len(named_majors) == 1 else set()


def matches_value(value: str, text: str, *, source_file: str = "", heading: str = "",
                  negated: bool = False) -> bool:
    """按数值边界匹配；带专业或字段的学分必须在对应语境中出现。"""
    value = normalize_value(value)
    if not value:
        return False
    majors, field = set(), ""
    credit = re.fullmatch(r"(.*?)(\d+(?:\.\d+)?学分)", value)
    if credit:
        majors, field = _credit_scope(credit[1])
        if majors or field:
            value = credit[2]

    pattern = re.escape(value)
    if value[0].isdigit():
        pattern = r"(?<!\d)(?<!\d\.)" + pattern
    if value[-1].isdigit():
        pattern += r"(?!\d|\.\d)"
    if value.endswith("元"):
        pattern += r"(?!每小时)"

    source_majors, _ = _credit_scope(source_file)
    _, source_field = _credit_scope(heading)
    supported_majors = set()
    pattern = re.compile(pattern)
    for sentence in _SENTENCE_BREAK.split(text):
        for clause, previous_majors in _scoped_clauses(sentence):
            for match in pattern.finditer(clause):
                prefix = clause[:match.start()]
                if bool(_NEGATION.search(prefix)) != negated:
                    continue
                if not (majors or field):
                    return True
                local_majors, local_field = _credit_scope(prefix)
                if field and field != (local_field or source_field):
                    continue
                if not majors:
                    return True
                supported_majors.update(local_majors or previous_majors or source_majors)
    return bool(majors) and majors <= supported_majors


def matches_condition(condition: str, text: str) -> bool:
    condition = normalize_value(condition)
    if not condition:
        return False
    pattern = re.escape(condition)
    if condition[-1].isdigit():
        pattern += r"(?!\d|\.\d)"
    for sentence in _SENTENCE_BREAK.split(_CITATION.sub("", text)):
        sentence = normalize_value(sentence)
        for match in re.finditer(pattern, sentence):
            if not _NEGATION.search(sentence[:match.start()]):
                return True
    return False


# ---------------------------------------------------------------------------
# 条件项抽取
# ---------------------------------------------------------------------------

# 条件关键词（教务政策高频条件）
_CONDITION_KEYWORDS = [
    "不及格", "挂科", "处分", "违纪", "贫困认定", "有效期",
    "前三位", "第一作者", "自主上报", "带公章", "公示", "通过资格审核",
    r"前\d+%", r"不低于\s*\d+", r"不超过\s*\d+",
]


def extract_conditions_from_text(text: str) -> list[str]:
    """从文本中抽取条件项。

    返回关键词及其简短上下文（去除引用标注 [N]）。
    """
    found = []
    # 先去除引用标注，避免干扰匹配
    clean_text = _CITATION.sub("", text)
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
    checks: list[ClaimCheck] = []
    for sentence in _SENTENCE_BREAK.split(answer):
        indices = {int(i) for i in _CITATION.findall(sentence)}
        docs = [doc for i, doc in enumerate(context_docs, 1) if not indices or i in indices]
        for clause, previous_majors in _scoped_clauses(sentence):
            for pattern, vtype in _VALUE_PATTERNS:
                for match in re.finditer(pattern, clause):
                    value = match[0]
                    prefix = clause[:match.start()]
                    majors, field = _credit_scope(prefix) if vtype == "credit" else (set(), "")
                    if vtype == "credit" and not majors:
                        majors = previous_majors
                    # 同一个数值可能适用于多个专业，每个专业都需要自己的证据。
                    matched = all(any(
                        matches_value(major + field + value, doc.page_content,
                                      source_file=doc.metadata.get("source_file", ""),
                                      heading=doc.metadata.get("heading_path", ""),
                                      negated=bool(_NEGATION.search(prefix)))
                        for doc in docs
                    ) for major in majors or {""})
                    checks.append(ClaimCheck(value, "value", matched))
            for cond in extract_conditions_from_text(clause):
                matched = any(matches_condition(cond, doc.page_content) for doc in docs)
                checks.append(ClaimCheck(cond, "condition", matched))

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
