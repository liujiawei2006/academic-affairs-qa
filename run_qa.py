"""教务问答入口：打通 检索 → 生成 最小链路。

用法：
    python run_qa.py

交互式输入问题，系统检索后调用 LLM 生成答案。
输入 q 退出。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.generation.answerer import Answerer
from src.generation.llm_client import LLMClient
from src.generation.verifier import format_report
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import RerankRetriever, SiliconFlowReranker
from src.vectorstore import load_vector_store


def build_retriever():
    """构建生产检索链路：Hybrid(RRF) + Rerank + 文档级去重。

    复用现有最终生产配置，确保答案生成基于最好的召回结果。
    """
    print("正在加载检索链路...")

    # 向量路
    vec_store = load_vector_store()
    vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})

    # BM25 路
    bm25_retriever = BM25Retriever()

    # 混合检索（RRF 融合）
    hybrid = HybridRetriever(vec_retriever, bm25_retriever, fusion="rrf", k=60)

    # Rerank + 文档级去重
    reranker = SiliconFlowReranker()
    retriever = RerankRetriever(
        hybrid, reranker,
        fetch_k=10,
        top_k=5,
        max_per_source=2,
    )

    print("检索链路加载完成")
    return retriever


def main():
    retriever = build_retriever()
    llm = LLMClient()
    answerer = Answerer(retriever, llm)

    print("\n" + "=" * 60)
    print("教务政策问答系统（DeepSeek 版）")
    print("输入问题开始对话，输入 q 退出")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if not question:
            continue

        result = answerer.answer(question)

        print(f"\n{'=' * 60}")
        print(f"问题：{result['question']}")
        print(f"{'=' * 60}")
        print(f"\n{result['answer']}")

        # 展示引用来源清单
        if result["citations"]:
            print(f"\n{'─' * 40}")
            print("引用来源：")
            for cit in result["citations"]:
                src = cit["source"]
                line = f"  {cit['marker']} 《{src.doc_title}》"
                if src.article:
                    line += f" {src.article}"
                elif src.chapter:
                    line += f" {src.chapter}"
                print(line)
                print(f"      文件：{src.source_file}")

        # 展示校验报告
        if result.get("verification"):
            vr = result["verification"]
            print(f"\n{'─' * 40}")
            print(format_report(vr))

        # 展示兜底状态
        fb = result.get("fallback")
        if fb and fb.should_fallback:
            print(f"\n{'─' * 40}")
            print(f"[兜底] {fb.reason}")


if __name__ == "__main__":
    main()
