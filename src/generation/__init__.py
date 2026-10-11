"""生成模块：LLM 客户端封装 + 答案生成 + 溯源标注 + 幻觉校验 + 无依据兜底。"""
from src.generation.answerer import Answerer
from src.generation.citation import CitationSource, format_citation_inline, parse_citations
from src.generation.fallback import FALLBACK_MESSAGE, FallbackDecision, check_fallback, pre_llm_fallback_check
from src.generation.llm_client import LLMClient
from src.generation.verifier import VerificationReport, format_report, verify

__all__ = [
    "Answerer",
    "CitationSource",
    "FallbackDecision",
    "FALLBACK_MESSAGE",
    "LLMClient",
    "VerificationReport",
    "check_fallback",
    "format_citation_inline",
    "format_report",
    "parse_citations",
    "pre_llm_fallback_check",
    "verify",
]
