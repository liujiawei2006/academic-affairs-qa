"""溯源数据结构与工具函数（B1 阶段产出）。

溯源字段规范：
    每条 context 来源应携带以下字段用于溯源标注：
    - source_file: 原始文件名（如 "2 浙工大发〔2022〕47号浙江工业大学本科学生奖学金评定及管理办法.pdf"）
    - doc_title: 从文件名提取的文档简称（如 "《浙江工业大学本科学生奖学金评定及管理办法》"）
    - chapter: 所属章（如 "第三章 个人荣誉称号及评定条件"）
    - heading_path: 最近一级标题路径
    - article: 条款编号（如 "第六条" "第十二条"），从原文 text 中提取
    - table_row: 表格行标识（针对表格 chunk），无则为空
    - chunk_id: 唯一分片标识

回复中的溯源标注格式：
    "依据《XX办法》第六条 [1]"
    其中 [1] 对应 context 中的编号，映射到具体来源。

答案中可能出现的标注模式：
    - 单条引用：……前 10%[1]。
    - 多条引用：……同时满足 [1][2]。
    - 带条款：依据《XX办法》第六条 [1]。
"""
import re
from dataclasses import dataclass, field


@dataclass
class CitationSource:
    """单条溯源来源（对应 context 中一段参考资料）。"""

    index: int  # context 中的编号（1-based）
    source_file: str  # 原始文件名
    doc_title: str  # 文档简称（用于回复中引用）
    chapter: str = ""  # 所属章
    heading_path: str = ""  # 标题路径
    article: str = ""  # 条款编号（如"第六条"）
    table_row: str = ""  # 表格行标识
    chunk_id: str = ""  # 分片唯一标识


# ---------------------------------------------------------------------------
# 文档标题提取
# ---------------------------------------------------------------------------

# 从文件名中提取文档简称的正则
_DOC_TITLE_PATTERNS = [
    # 匹配 "浙江工业大学XXX管理办法" / "XXX培养计划" / "XXX通知" 等
    r"(浙江工业大学.+?(?:管理办法|评定办法|实施办法|申请操作|招募通知|提交指南|补充说明|认定办法|命题范围|名额分布))",
    # 匹配 "XXX培养计划"
    r"(\d{4}.+?培养计划)",
    # 匹配 "XXX通知"
    r"(关于.+?通知)",
]


def extract_doc_title(source_file: str) -> str:
    """从文件名提取可读的文档简称。

    规则：
    1. 优先匹配"浙江工业大学XXX办法"等正式名称
    2. 去掉文件编号前缀（如 "1 浙工大发〔2022〕46号"）
    3. 去掉文件扩展名
    4. 无法匹配时返回去掉编号后的文件名
    """
    # 去掉扩展名
    name = re.sub(r"\.(pdf|docx?|txt)$", "", source_file, flags=re.IGNORECASE)

    # 去掉前导编号（如 "1 ", "2 ", "7 "）和发文字号（如 "浙工大发〔2022〕46号"）
    name = re.sub(r"^\d+\s*", "", name)
    name = re.sub(r"浙工大发[〔\[]\d{4}[〕\]]\d+号", "", name)

    # 尝试匹配文档标题
    for pattern in _DOC_TITLE_PATTERNS:
        m = re.search(pattern, name)
        if m:
            return m.group(1)

    # fallback：返回清理后的文件名
    name = name.strip()
    return name if name else source_file


# ---------------------------------------------------------------------------
# 条款编号提取
# ---------------------------------------------------------------------------

# 中文条款编号正则："第X条" / "第X章" / "第X节"
_ARTICLE_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千\d]+[条章节])"
)


def extract_article_from_text(text: str) -> str:
    """从 chunk 文本中提取第一个条款编号。

    优先级：条 > 章 > 节。
    返回如 "第六条"，找不到则返回空字符串。
    """
    matches = _ARTICLE_PATTERN.findall(text)
    if not matches:
        return ""

    # 优先返回"第X条"（最细粒度）
    for m in matches:
        if "条" in m:
            return m
    # 其次"第X章"
    for m in matches:
        if "章" in m:
            return m
    return matches[0]


def extract_chapter_from_text(text: str) -> str:
    """从 chunk 文本中提取章标题（如"第三章 个人荣誉称号及评定条件"）。"""
    m = re.search(r"(第[一二三四五六七八九十百千\d]+章\s*\S+)", text)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# 溯源标注后处理
# ---------------------------------------------------------------------------

# 匹配 LLM 回复中的 [数字] 引用标注
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def parse_citations(answer_text: str, sources: list[CitationSource]) -> list[dict]:
    """解析回复中的引用标注，映射回真实来源。

    返回引用列表，每条包含：
        {"marker": "[1]", "source": CitationSource}

    仅保留实际出现在回复中且编号有效的引用。
    """
    cited_indices = set(int(m) for m in _CITATION_PATTERN.findall(answer_text))
    results = []
    for src in sources:
        if src.index in cited_indices:
            results.append({"marker": f"[{src.index}]", "source": src})
    return results


def format_citation_inline(source: CitationSource) -> str:
    """生成单条溯源的内联展示文本。

    格式：依据《doc_title》article/chapter [index]
    示例：依据《浙江工业大学本科学生奖学金评定及管理办法》第六条 [1]
    """
    parts = [f"《{source.doc_title}》"]
    if source.article:
        parts.append(source.article)
    elif source.chapter:
        parts.append(source.chapter)
    parts.append(f"[{source.index}]")
    return "依据" + "".join(parts)
