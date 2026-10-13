"""FastAPI 应用：封装检索 + 生成为 HTTP 接口。

启动方式：
    python run_api.py
    # 或
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

接口：
    POST /qa         问答接口（JSON 响应）
    POST /qa/stream  问答接口（SSE 流式输出）
    GET  /health     健康检查
"""
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.generation.answerer import Answerer
from src.generation.citation import CitationSource
from src.generation.fallback import FallbackDecision
from src.generation.llm_client import LLMClient
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import RerankRetriever, SiliconFlowReranker
from src.vectorstore import load_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局单例（lifespan 中初始化）
# ---------------------------------------------------------------------------
_answerer: Answerer | None = None
_llm_client: LLMClient | None = None
_vec_store = None  # Milvus 向量库实例，供 /health 探活


def _build_retriever():
    """构建生产检索链路：Hybrid(RRF) + Rerank + 文档级去重。"""
    global _vec_store
    _vec_store = load_vector_store()
    vec_retriever = _vec_store.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever()
    hybrid = HybridRetriever(vec_retriever, bm25_retriever, fusion="rrf", k=60)
    reranker = SiliconFlowReranker()
    return RerankRetriever(hybrid, reranker, fetch_k=10, top_k=5, max_per_source=2)


# ---------------------------------------------------------------------------
# Lifespan：启动时加载检索链路，关闭时清理
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化全局检索 + 生成组件。"""
    global _answerer, _llm_client
    logger.info("正在初始化检索链路...")
    # 检索链路构建是同步阻塞操作，放到线程池执行
    loop = asyncio.get_event_loop()
    retriever = await loop.run_in_executor(None, _build_retriever)
    _llm_client = LLMClient()
    _answerer = Answerer(retriever, _llm_client)
    logger.info("检索链路初始化完成")
    yield
    _answerer = None
    _llm_client = None
    logger.info("应用已关闭")


# ---------------------------------------------------------------------------
# FastAPI 实例
# ---------------------------------------------------------------------------
app = FastAPI(
    title="教务政策问答系统",
    description="基于 RAG 的高校教务政策知识库问答 API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 全链路日志中间件：request_id + 请求耗时
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """全链路中间件：request_id + 请求耗时 + 全局异常捕获。"""
    request_id = str(uuid.uuid4())[:8]  # 取前 8 位足够标识
    request.state.request_id = request_id

    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        elapsed = round(time.perf_counter() - start, 3)

        # 在响应头中携带 request_id 和耗时，方便调试
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed}s"

        logger.info(
            "[%s] %s %s → %s (%.3fs)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        logger.error(
            "[%s] %s %s 异常: %s (%.3fs)",
            request_id,
            request.method,
            request.url.path,
            e,
            elapsed,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "服务器内部错误",
                "detail": str(e),
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class QARequest(BaseModel):
    """问答请求。"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    enable_verify: bool = Field(True, description="是否启用原文事实校验")


class CitationItem(BaseModel):
    """单条引用来源。"""
    marker: str = Field(..., description="引用编号，如 [1]")
    source_file: str = Field("", description="原始文件名")
    doc_title: str = Field("", description="文档简称")
    chapter: str = Field("", description="所属章")
    article: str = Field("", description="条款编号")
    table_row: str = Field("", description="表格行标识")


class FallbackInfo(BaseModel):
    """兜底信息。"""
    should_fallback: bool
    reason: str = ""


class TimingInfo(BaseModel):
    """各阶段耗时（秒）。"""
    retrieval_s: float = Field(0, description="检索耗时")
    context_build_s: float = Field(0, description="Context 构建耗时")
    llm_s: float = Field(0, description="LLM 生成耗时")
    verify_s: float = Field(0, description="事实校验耗时")


class TokenUsage(BaseModel):
    """Token 消耗统计。"""
    prompt_tokens: int = Field(0, description="输入 Token 数")
    completion_tokens: int = Field(0, description="输出 Token 数")
    total_tokens: int = Field(0, description="总 Token 数")
    call_count: int = Field(0, description="LLM 调用次数")


class QAResponse(BaseModel):
    """问答响应。"""
    question: str
    answer: str
    citations: list[CitationItem] = []
    fallback: FallbackInfo = FallbackInfo(should_fallback=False)
    timing: TimingInfo = Field(default_factory=TimingInfo, description="各阶段耗时")
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Token 消耗")


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------
def _serialize_citation(cit: dict) -> CitationItem:
    """将 answerer 返回的 citation dict 序列化为 Pydantic 模型。"""
    source: CitationSource = cit["source"]
    return CitationItem(
        marker=cit["marker"],
        source_file=source.source_file,
        doc_title=source.doc_title,
        chapter=source.chapter,
        article=source.article,
        table_row=source.table_row,
    )


def _serialize_fallback(fb: FallbackDecision) -> FallbackInfo:
    """将 FallbackDecision 序列化为 Pydantic 模型。"""
    return FallbackInfo(
        should_fallback=fb.should_fallback,
        reason=fb.reason,
    )


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/health", tags=["系统"])
async def health():
    """健康检查：探活 Milvus 向量库 + LLM API 配置。"""
    checks = {}

    # 1. Milvus 探活：尝试执行轻量操作
    try:
        if _vec_store is not None:
            # 通过 collection 属性检查连接是否正常
            _ = _vec_store.collection_name
            checks["milvus"] = True
        else:
            checks["milvus"] = False
    except Exception as e:
        logger.warning("Milvus 探活失败: %s", e)
        checks["milvus"] = False

    # 2. LLM 探活：检查 API Key 是否已配置（不做实际 API 调用，避免成本）
    api_key = os.getenv("LLM_API_KEY", "")
    checks["llm"] = bool(api_key and api_key != "sk-你的密钥")

    # 3. 整体状态
    all_ok = all(checks.values())
    status = "ok" if all_ok else "degraded"

    return {"status": status, "checks": checks}


@app.post("/qa", response_model=QAResponse, tags=["问答"])
async def qa(req: QARequest, request: Request):
    """问答接口：输入问题，返回答案 + 引用来源。"""
    if _answerer is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    request_id = getattr(request.state, "request_id", "-")
    logger.info("[%s] 收到问题: %s", request_id, req.question[:50])

    try:
        # Answerer.answer() 是同步阻塞调用，放到线程池避免阻塞事件循环
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None, _answerer.answer, req.question, 5, req.enable_verify,
        )
        result = await asyncio.wait_for(future, timeout=120)
    except asyncio.TimeoutError:
        future.cancel()  # 标记取消（线程仍在运行，但防止结果回写）
        logger.error("[%s] 问答超时（120s），线程已标记取消: %s", request_id, req.question[:50])
        raise HTTPException(status_code=504, detail="回答超时，请稍后重试")
    except Exception as e:
        logger.error("[%s] 问答异常: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"处理异常: {e}")

    # 序列化响应
    citations = [_serialize_citation(c) for c in result.get("citations", [])]
    fallback = _serialize_fallback(result.get("fallback", FallbackDecision(False)))
    timing = TimingInfo(**result.get("timing", {}))
    token_usage = TokenUsage(**result.get("token_usage", {}))

    logger.info(
        "[%s] 回答完成: 引用=%d, 耗时=%.3fs, Token=%d",
        request_id,
        len(citations),
        sum([timing.retrieval_s, timing.context_build_s, timing.llm_s, timing.verify_s]),
        token_usage.total_tokens,
    )

    return QAResponse(
        question=result["question"],
        answer=result["answer"],
        citations=citations,
        fallback=fallback,
        timing=timing,
        token_usage=token_usage,
    )


@app.post("/qa/stream", tags=["问答"])
async def qa_stream(req: QARequest, request: Request):
    """流式问答接口：SSE 逐 token 推送答案，最后推送元数据。

    SSE 事件类型：
        event: token     data: {"content": "..."}        每个文本片段
        event: fallback  data: {"answer": "...", "reason": "..."}  兜底
        event: done      data: {"answer": "...", "citations": [...], ...}  流结束元数据
    """
    if _answerer is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    request_id = getattr(request.state, "request_id", "-")
    logger.info("[%s] 流式请求: %s", request_id, req.question[:50])

    async def event_generator():
        """SSE 事件生成器：将 Answerer 的流式事件转换为 SSE 格式。"""
        try:
            async with asyncio.timeout(120):  # 全局超时 120s，与 /qa 一致
                async for event in _answerer.astream_answer(req.question, top_k=5):
                    event_type = event["type"]

                    if event_type == "token":
                        yield {
                            "event": "token",
                            "data": json.dumps({"content": event["content"]}, ensure_ascii=False),
                        }

                    elif event_type == "fallback":
                        yield {
                            "event": "fallback",
                            "data": json.dumps(
                                {"answer": event["answer"], "reason": event["reason"]},
                                ensure_ascii=False,
                            ),
                        }

                    elif event_type == "done":
                        # 序列化 citations
                        citations = [_serialize_citation(c) for c in event.get("citations", [])]
                        timing = TimingInfo(**event.get("timing", {}))
                        token_usage = TokenUsage(**event.get("token_usage", {}))

                        logger.info(
                            "[%s] 流式完成: 引用=%d, Token=%d",
                            request_id,
                            len(citations),
                            token_usage.total_tokens,
                        )

                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {
                                    "answer": event["answer"],
                                    "citations": [c.model_dump() for c in citations],
                                    "timing": timing.model_dump(),
                                    "token_usage": token_usage.model_dump(),
                                },
                                ensure_ascii=False,
                            ),
                        }
        except TimeoutError:
            logger.error("[%s] 流式输出超时（120s）", request_id)
            yield {
                "event": "error",
                "data": json.dumps({"error": "回答超时，请稍后重试"}, ensure_ascii=False),
            }
        except Exception as e:
            logger.error("[%s] 流式输出异常: %s", request_id, e)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
