"""3 组消融实验(99 题): vector / hybrid / rewrite。

对比 vector / hybrid / rewrite 三种检索策略的召回率。
每组配置独立跑 evaluate_recall 并落盘到 docs/eval_results/<label>/。

使用方式:
    python run_ablation_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import evaluate_recall, load_golden_dataset, _save_to_eval_results
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.vectorstore import load_vector_store
from src.query.rewriter import (
    CachedLLMRewriter,
    RewriteRetriever,
)
from run_hybrid_eval import ScoredVectorRetriever


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    dataset = load_golden_dataset("tests/golden_dataset.jsonl")
    print(f"加载测试集: {len(dataset)} 条")

    print("加载向量库...")
    vectorstore = load_vector_store()

    print("构建 BM25 检索器...")
    bm25 = BM25Retriever(top_k=10)

    print("构建 HybridRetriever (weighted_05_05)...")
    vec = ScoredVectorRetriever(vectorstore, k=10)
    hybrid = HybridRetriever(
        vec, bm25, fusion="weighted", weights=[0.5, 0.5], top_k=10,
    )

    print("加载 LLM 改写器(复用改写缓存)...")
    rewriter = CachedLLMRewriter()

    # 3 组配置
    configs = {
        "vector": vectorstore.as_retriever(search_kwargs={"k": 10}),
        "hybrid": hybrid,
        "rewrite": RewriteRetriever(hybrid, rewriter),
    }

    results = {}
    for label, retriever in configs.items():
        print(f"\n{'=' * 60}\n运行配置: {label}\n{'=' * 60}")
        r = evaluate_recall(
            retriever, dataset, top_k_list=[1, 3, 5], max_fetch_k=10,
        )
        report = {
            "label": label,
            "recall": r["summary"],
            "per_question": r["per_question"],
            "total_questions": r["total_questions"],
            "table_questions": r["table_questions"],
        }
        _save_to_eval_results(report, label, "eval_results")
        results[label] = r["summary"]
        print(
            f"  Top1={r['summary'].get('top1_recall', 0):.1%}  "
            f"Top3={r['summary'].get('top3_recall', 0):.1%}  "
            f"Top5={r['summary'].get('top5_recall', 0):.1%}  "
            f"表格Top1={r['summary'].get('top1_table_recall', 0):.1%}"
        )

    # 汇总表
    print("\n" + "=" * 90)
    print("   3 组消融实验对比")
    print("=" * 90)
    print(
        f"{'配置':<15} {'Top1':>8} {'Top3':>8} {'Top5':>8} "
        f"{'表格Top1':>10} {'表格Top3':>10}"
    )
    print("-" * 90)
    for label, s in results.items():
        print(
            f"{label:<15} "
            f"{s.get('top1_recall', 0):>7.1%} "
            f"{s.get('top3_recall', 0):>7.1%} "
            f"{s.get('top5_recall', 0):>7.1%} "
            f"{s.get('top1_table_recall', 0):>9.1%} "
            f"{s.get('top3_table_recall', 0):>9.1%}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
