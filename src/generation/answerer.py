"""答案生成模块：检索 → 拼 context → LLM 生成答案（含溯源标注）。

B2 阶段：启用溯源标注 prompt，模型每个结论必须标注 [N] 引用编号，
后处理将编号映射回真实文件名 + 条款。

数据流：
    question → retriever.invoke() → top-k Document
    → _build_citation_sources() 构建溯源元数据
    → _build_context() 拼成带编号的参考资料
    → 溯源版 prompt → LLM 生成
    → _resolve_citations() 后处理映射
    → 返回 {question, answer, citations, sources, context_docs}
"""
import asyncio
import re
import time

from langchain_core.documents import Document

from src.generation.citation import (
    CitationSource,
    extract_article_from_text,
    extract_chapter_from_text,
    extract_doc_title,
    format_citation_inline,
    parse_citations,
)
from src.generation.fallback import (
    FALLBACK_MESSAGE,
    FallbackDecision,
    check_fallback,
    pre_llm_fallback_check,
)
from src.generation.llm_client import LLMClient
from src.generation.verifier import VerificationReport, verify


# C2 约束型 prompt：条件句逐条核对 + 表格数值原样引用 + 兜底转接
QA_PROMPT_TEMPLATE = """\
你是高校教务政策知识库问答助手。请根据以下编号参考资料回答学生的问题。

【强制约束（违反任何一条即为幻觉）】

1. 事实来源约束
   - 只能依据参考资料中的内容回答，禁止编造任何信息
   - 每个事实性结论后必须紧跟引用编号 [N]，N 是参考资料编号
   - 引用优先写完整格式："依据《文档名》第X条 [N]"

2. 条件句逐条核对（核心：教务政策多为"满足以下全部条件方可…"句式）
   - 当参考资料中出现编号列表（1）（2）…或①②③…的并列条件时，必须逐条列出，不得遗漏
   - 并列条件（"同时具备""且""均"）必须全部列出，漏一条即为幻觉
   - 排除条件（"取消资格""不具备""不得"）必须明确说明
   - 示例：如果原文有"①绩点不低于3.8 ②综合测评前10% ③无不及格课程"，回答必须三条都写

3. 表格数值原样引用
   - 参考资料中的具体数值（绩点、百分比、金额、学分、人数等）必须原样引用，禁止计算、推断或四舍五入
   - 例如：原文写"3.8"就不能写"3.8以上"；原文写"25元每小时"就不能写"25元/小时"
   - 涉及多个数值的对比题，每个数值必须分别标注来源

4. 兜底转接
   - 如果参考资料中没有相关信息，如实说明："未在现行政策中找到依据，建议咨询教务处"
   - 部分有依据、部分无依据时，有依据的部分正常回答并标注来源，无依据的部分明确说明"以下内容未在现行政策中找到依据"

参考资料：
{context}

学生问题：{question}

回答（每条结论必须标注 [N] 来源编号，条件句必须逐条列出）："""


class Answerer:
    """答案生成器：检索 → 拼 context → LLM 生成 → 溯源解析。

    用法：
        from src.generation import Answerer
        # ... 构建 retriever ...
        answerer = Answerer(retriever)
        result = answerer.answer("一等奖学金绩点要求是多少？")
        print(result["answer"])      # 含 [N] 标注的答案
        print(result["citations"])   # 解析后的引用映射
    """

    def __init__(self, retriever, llm_client: LLMClient | None = None):
        """
        参数：
            retriever: 任意支持 .invoke(query) → list[Document] 的检索器
            llm_client: LLM 客户端实例，默认新建一个
        """
        self.retriever = retriever
        self.llm = llm_client or LLMClient()

    def answer(self, question: str, top_k: int = 5, enable_verify: bool = True) -> dict:
        """检索 + 生成带溯源标注的答案 + （可选）原文事实校验 + 兜底判定。

        参数：
            question: 用户问题
            top_k: 检索返回条数
            enable_verify: 是否启用原文事实校验（默认开启）

        返回：
            {
                "question": str,
                "answer": str,              # 含 [N] 引用的原始回复，或兜底话术
                "citations": list[dict],     # 解析后的引用映射
                "sources": list[CitationSource],  # 全部来源列表
                "context_docs": list[Document],
                "verification": VerificationReport | None,  # 校验报告
                "fallback": FallbackDecision,  # 兜底判定结果
                "timing": dict,              # 各阶段耗时（秒）
                "token_usage": dict,         # Token 消耗统计
            }
        """
        timing = {}  # 各阶段耗时记录

        # 1. 检索
        t0 = time.perf_counter()
        docs = self.retriever.invoke(question)
        top_docs = docs[:top_k]
        timing["retrieval_s"] = round(time.perf_counter() - t0, 3)

        # 2. 前置兜底检查（召回为空 + 超范围问题）→ 直接返回，节省 LLM 调用
        pre_fallback = pre_llm_fallback_check(question, top_docs)
        if pre_fallback.should_fallback:
            result = self._build_fallback_result(question, top_docs, pre_fallback)
            result["timing"] = timing
            # 前置兜底未调用 LLM，Token 消耗为 0
            result["token_usage"] = {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "call_count": 0,
            }
            return result

        # 3. 构建溯源元数据 + 拼 context
        t0 = time.perf_counter()
        sources = self._build_citation_sources(top_docs)
        context = self._build_context(top_docs, sources)
        timing["context_build_s"] = round(time.perf_counter() - t0, 3)

        # 4. LLM 生成（记录 Token 消耗前后差值）
        t0 = time.perf_counter()
        self.llm.reset_token_stats()  # 重置为本次请求粒度
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        answer_text = self.llm.invoke(prompt)
        timing["llm_s"] = round(time.perf_counter() - t0, 3)
        token_usage = self.llm.get_token_stats()  # 本次请求的 Token 消耗

        # 5. 后处理：解析引用标注 → 映射回来源
        citations = parse_citations(answer_text, sources)

        # 6. 原文事实校验（C3）
        verification = None
        if enable_verify:
            t0 = time.perf_counter()
            verification = verify(answer_text, top_docs)
            timing["verify_s"] = round(time.perf_counter() - t0, 3)

        # 7. 后置兜底检查（C3 校验幻觉率过高）→ 覆盖答案
        post_fallback = check_fallback(question, top_docs, verification)
        if post_fallback.should_fallback:
            answer_text = FALLBACK_MESSAGE
            citations = []  # 兜底时无引用

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "sources": sources,
            "context_docs": top_docs,
            "verification": verification,
            "fallback": post_fallback if post_fallback.should_fallback else pre_fallback,
            "timing": timing,
            "token_usage": token_usage,
        }

    @staticmethod
    def _build_fallback_result(
        question: str,
        docs: list[Document],
        fallback: FallbackDecision,
    ) -> dict:
        """构建兜底结果（不调用 LLM）。"""
        sources = Answerer._build_citation_sources(docs)
        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "citations": [],
            "sources": sources,
            "context_docs": docs,
            "verification": None,
            "fallback": fallback,
        }

    @staticmethod
    def _build_citation_sources(docs: list[Document]) -> list[CitationSource]:
        """从检索到的 Document 列表构建 CitationSource 列表。"""
        sources = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            source_file = meta.get("source_file", "")
            text = doc.page_content or ""

            # 提取文档简称
            doc_title = extract_doc_title(source_file)

            # 提取条款编号（优先从文本中找“第X条”）
            article = extract_article_from_text(text)

            # 提取章信息（优先 metadata，其次从文本提取）
            chapter = meta.get("chapter", "") or extract_chapter_from_text(text)

            # 表格行标识（表格 chunk 的 heading_path 通常含具体行信息）
            heading_path = meta.get("heading_path", "")
            table_row = ""
            if meta.get("chunk_strategy") == "table":
                table_row = heading_path

            sources.append(CitationSource(
                index=i,
                source_file=source_file,
                doc_title=doc_title,
                chapter=chapter,
                heading_path=heading_path,
                article=article,
                table_row=table_row,
                chunk_id=meta.get("chunk_id", ""),
            ))
        return sources

    @staticmethod
    def _build_context(docs: list[Document], sources: list[CitationSource]) -> str:
        """将检索到的文档拼成带编号的参考资料。

        格式（每条 context 头部显示来源信息，方便模型引用）：
            [1] 来源：《浙江工业大学本科学生奖学金评定及管理办法》第六条
            正文内容...

            ---

            [2] 来源：《XXX》第三章
            正文内容...
        """
        if not docs:
            return "（无相关参考资料）"

        parts = []
        for doc, src in zip(docs, sources):
            # 构造简洁的来源描述行
            source_desc = f"《{src.doc_title}》"
            if src.article:
                source_desc += f" {src.article}"
            elif src.chapter:
                source_desc += f" {src.chapter}"
            if src.table_row:
                source_desc += f" | 表格：{src.table_row}"

            header = f"[{src.index}] 来源：{source_desc}"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def format_answer_with_citation_list(result: dict) -> str:
        """将答案 + 引用来源清单格式化为可读文本。

        用于 run_qa.py 的终端输出展示。
        """
        lines = [result["answer"], "", "─" * 40, "引用来源："]
        for cit in result["citations"]:
            src = cit["source"]
            lines.append(f"  {cit['marker']} {format_citation_inline(src)}")
        return "\n".join(lines)

    async def astream_answer(
        self, question: str, top_k: int = 5,
    ):
        """流式版 answer()：检索同步执行，LLM 生成逐 token 异步输出。

        异步生成器，yield dict 事件：
            {"type": "token", "content": str}       每个文本片段
            {"type": "fallback", "answer": str, "reason": str}  兜底（无 LLM 调用）
            {"type": "done", "answer": str, "citations": list,
             "timing": dict, "token_usage": dict}  流结束元数据

        用法：
            async for event in answerer.astream_answer("问题"):
                if event["type"] == "token":
                    print(event["content"], end="")
                elif event["type"] == "done":
                    print(event["citations"])
        """
        timing = {}

        # 1. 检索（同步阻塞，放线程池）
        t0 = time.perf_counter()
        docs = await asyncio.to_thread(self.retriever.invoke, question)
        top_docs = docs[:top_k]
        timing["retrieval_s"] = round(time.perf_counter() - t0, 3)

        # 2. 前置兜底检查
        pre_fallback = pre_llm_fallback_check(question, top_docs)
        if pre_fallback.should_fallback:
            yield {
                "type": "fallback",
                "answer": FALLBACK_MESSAGE,
                "reason": pre_fallback.reason,
            }
            return

        # 3. 构建溯源元数据 + 拼 context
        t0 = time.perf_counter()
        sources = self._build_citation_sources(top_docs)
        context = self._build_context(top_docs, sources)
        timing["context_build_s"] = round(time.perf_counter() - t0, 3)

        # 4. LLM 流式生成（用差值法避免并发请求的 Token 统计竞态）
        t0 = time.perf_counter()
        stats_before = self.llm.get_token_stats()
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        answer_parts = []
        async for chunk in self.llm.astream(prompt):
            answer_parts.append(chunk)
            yield {"type": "token", "content": chunk}

        timing["llm_s"] = round(time.perf_counter() - t0, 3)
        # 差值计算：本次请求的 Token 消耗 = 流后统计 - 流前统计
        stats_after = self.llm.get_token_stats()
        token_usage = {
            "prompt_tokens": stats_after["prompt_tokens"] - stats_before["prompt_tokens"],
            "completion_tokens": stats_after["completion_tokens"] - stats_before["completion_tokens"],
            "total_tokens": stats_after["total_tokens"] - stats_before["total_tokens"],
            "call_count": stats_after["call_count"] - stats_before["call_count"],
        }

        # 5. 后处理：解析引用标注
        answer_text = "".join(answer_parts)
        citations = parse_citations(answer_text, sources)

        yield {
            "type": "done",
            "answer": answer_text,
            "citations": citations,
            "timing": timing,
            "token_usage": token_usage,
        }
