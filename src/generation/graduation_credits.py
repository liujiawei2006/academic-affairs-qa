"""从培养计划的明确条款回答单个专业的毕业总学分。"""
import re

from langchain_core.documents import Document


_TOTAL_QUESTION = re.compile(
    r"(?:请问)?(?:(?P<year>20\d{2})(?:年级|级|年|版)?的?)?"
    r"(?P<major>[\u4e00-\u9fff]+?)(?:专业)?的?毕业"
    r"(?:(?:总)?学分(?:要求)?(?:是|为|有)?(?:多少|几)(?:学分)?"
    r"|(?:一共|总共|至少)?(?:需要|需|要|须)?(?:至少)?"
    r"(?:修满|修够|修|达到)?(?:多少|几)学分)[？?。！!]*"
)
_PLAN_FILE = re.compile(r"(?P<year>20\d{2})\s*(?P<major>.+?)(?:专业)?培养(?:计划|方案)\.pdf")
_TOTAL_FIELD = re.compile(r"毕业(?:总)?学分要求\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*学分")


def lookup_graduation_total(question: str, chunks: list[dict]) -> tuple[str, list[Document]] | None:
    query = _TOTAL_QUESTION.fullmatch(re.sub(r"\s+", "", question))
    if not query:
        return None

    major, year = query["major"], query["year"]
    plans = {}
    for chunk in chunks:
        plan = _PLAN_FILE.fullmatch(chunk["source_file"])
        if plan and plan["major"] == major:
            plans.setdefault((plan["year"], chunk["source_file"]), []).append(chunk)
    if not plans:
        return None

    candidates = [(key, docs) for key, docs in plans.items() if not year or key[0] == year]
    if len(candidates) != 1:
        return "", []

    (year, source_file), plan_chunks = candidates[0]
    evidence = {}
    for chunk in plan_chunks:
        for credits in _TOTAL_FIELD.findall(chunk["text"]):
            evidence.setdefault(credits, chunk)
    if len(evidence) != 1:
        return "", []

    credits, chunk = next(iter(evidence.items()))
    metadata = dict(chunk.get("metadata", {}))
    metadata.update(source_file=source_file, chunk_id=chunk.get("chunk_id", ""))
    if metadata.get("heading_path"):
        metadata["chapter"] = metadata["heading_path"]
    doc = Document(page_content=chunk["text"], metadata=metadata)
    answer = f"按 {year} 版培养计划，{major}专业毕业总学分为 {credits} 学分 [1]。"
    return answer, [doc]
