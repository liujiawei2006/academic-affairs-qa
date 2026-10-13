"""FastAPI 应用：封装检索 + 生成为 HTTP 接口。

启动方式：
    python run_api.py
    # 或
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

接口：
    POST /qa      问答接口（JSON 响应）
    GET  /health  健康检查
"""
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

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


def _build_retriever():
    """构建生产检索链路：Hybrid(RRF) + Rerank + 文档级去重。"""
    vec_store = load_vector_store()
    vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})
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
    global _answerer
    logger.info("正在初始化检索链路...")
    # 检索链路构建是同步阻塞操作，放到线程池执行
    loop = asyncio.get_event_loop()
    retriever = await loop.run_in_executor(None, _build_retriever)
    llm = LLMClient()
    _answerer = Answerer(retriever, llm)
    logger.info("检索链路初始化完成")
    yield
    _answerer = None
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
    """为每个请求生成唯一 request_id，记录总耗时。"""
    request_id = str(uuid.uuid4())[:8]  # 取前 8 位足够标识
    request.state.request_id = request_id

    start = time.perf_counter()
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
    """健康检查。"""
    return {"status": "ok"}


@app.post("/qa", response_model=QAResponse, tags=["问答"])
async def qa(req: QARequest, request: Request):
    """问答接口：输入问题，返回答案 + 引用来源。"""
    if _answerer is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    request_id = getattr(request.state, "request_id", "-")
    logger.info("[%s] 收到问题: %s", request_id, req.question[:50])

    # Answerer.answer() 是同步阻塞调用，放到线程池避免阻塞事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _answerer.answer, req.question, 5, req.enable_verify)

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
