"""请求历史存储：内存中保留最近 N 条请求记录，供看板展示。

设计：
- 线程安全的环形缓冲区（collections.deque + maxlen）
- 每条记录包含：request_id、question、召回文档、耗时、Token、引用数等
- 供 /dashboard 看板和 /api/history API 使用
"""
import logging
from collections import deque
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

# 最大保留记录数
MAX_HISTORY = 100

# 全局存储（线程安全）
_history: deque = deque(maxlen=MAX_HISTORY)
_lock = Lock()


def record_request(record: dict):
    """记录一条请求到历史。

    参数：
        record: 包含以下字段的 dict：
            - request_id: str
            - question: str
            - timestamp: str (ISO 格式)
            - total_s: float (总耗时)
            - retrieval_s: float
            - llm_s: float
            - total_tokens: int
            - citation_count: int
            - fallback: bool
            - retrieved_docs: list[dict] (召回文档列表)
    """
    with _lock:
        _history.append(record)
    logger.debug("记录请求历史: %s (%s)", record["request_id"], record["question"][:30])


def get_history(limit: int = 50) -> list[dict]:
    """获取最近的请求历史。

    参数：
        limit: 返回条数（默认 50，最大 MAX_HISTORY）

    返回：
        按时间倒序的请求记录列表
    """
    with _lock:
        items = list(_history)
    # 倒序返回（最新的在前）
    items.reverse()
    return items[:min(limit, MAX_HISTORY)]


def get_stats() -> dict:
    """获取汇总统计。"""
    with _lock:
        items = list(_history)

    if not items:
        return {
            "total_requests": 0,
            "avg_total_s": 0,
            "avg_retrieval_s": 0,
            "avg_llm_s": 0,
            "avg_tokens": 0,
            "avg_citations": 0,
            "fallback_count": 0,
            "fallback_rate": 0,
        }

    n = len(items)
    return {
        "total_requests": n,
        "avg_total_s": round(sum(r["total_s"] for r in items) / n, 3),
        "avg_retrieval_s": round(sum(r["retrieval_s"] for r in items) / n, 3),
        "avg_llm_s": round(sum(r["llm_s"] for r in items) / n, 3),
        "avg_tokens": round(sum(r["total_tokens"] for r in items) / n),
        "avg_citations": round(sum(r["citation_count"] for r in items) / n, 1),
        "fallback_count": sum(1 for r in items if r["fallback"]),
        "fallback_rate": round(sum(1 for r in items if r["fallback"]) / n * 100, 1),
    }
