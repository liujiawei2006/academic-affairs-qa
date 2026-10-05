"""阶段 3 增量评估：改写 vs 改写+Rerank。

对比实验：
- rewrite（阶段 2 最优配置，对照组）
- rewrite + rerank（SiliconFlow Rerank API 精排，实验组）

产出物：
- docs/eval_results/rerank/{result.json, report.md}
- docs/eval_results/rerank_comparison.json（逐题排名对比）
- 控制台输出对比表 + rerank 前后排名明细

使用方式：
    python run_rerank_eval.py
    python run_rerank_eval.py --fetch-k 20 --top-k 5
    python run_rerank_eval.py --dataset tests/golden_dataset.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import evaluate_recall, load_golden_dataset, _save_to_eval_results
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import RerankRetriever, SiliconFlowReranker
from src.vectorstore import load_vector_store
from src.query.rewriter import CachedLLMRewriter, RewriteRetriever


# ---------------------------------------------------------------------------
# 带分数的向量检索器封装（加权融合需要分数）
# ---------------------------------------------------------------------------
class ScoredVectorRetriever:
    """包装 Milvus retriever，让返回的 Document 带 score 字段。"""

    def __init__(self, vectorstore, k: int = 10):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query: str) -> list:
        from langchain_core.documents import Document

        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.k
            )
            results = []
            for doc, distance in docs_and_scores:
                doc.metadata["score"] = 1.0 - float(distance)
                results.append(doc)
            return results
        except Exception as e:
            print(f"  带分数检索失败: {e}，回退到无分数模式")
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            )
            return retriever.invoke(query)


# ---------------------------------------------------------------------------
# 逐题排名对比
# ---------------------------------------------------------------------------
def build_comparison(
    rewrite_result: dict,
    rerank_result: dict,
) -> list[dict]:
    """生成逐题对比明细：rerank 前排名 → rerank 后排名。"""
    rewrite_per_q = {p["id"]: p for p in rewrite_result.get("per_question", [])}
    rerank_per_q = {p["id"]: p for p in rerank_result.get("per_question", [])}

    comparison = []
    for qid in sorted(rewrite_per_q.keys()):
        rw = rewrite_per_q[qid]
        rr = rerank_per_q.get(qid, {})

        expected = set(rw.get("expected_source_files", []))
        rw_rank = _find_rank(rw, expected)
        rr_rank = _find_rank(rr, expected)

        rw_hit1 = rw.get("hit", {}).get("1", False)
        rr_hit1 = rr.get("hit", {}).get("1", False)
        rw_hit3 = rw.get("hit", {}).get("3", False)
        rr_hit3 = rr.get("hit", {}).get("3", False)

        # 变化分类
        if rr_hit1 and not rw_hit1:
            change = "improved_top1"
        elif rr_hit3 and not rw_hit3:
            change = "improved_top3"
        elif rw_hit1 and not rr_hit1:
            change = "regressed_top1"
        elif rw_hit3 and not rr_hit3:
            change = "regressed_top3"
        else:
            change = "unchanged"

        comparison.append({
            "id": qid,
            "question": rw["question"],
            "is_table_question": rw.get("is_table_question", False),
            "rewrite_top1": rw_hit1,
            "rerank_top1": rr_hit1,
            "rewrite_top3": rw_hit3,
            "rerank_top3": rr_hit3,
            "rewrite_rank": rw_rank,
            "rerank_rank": rr_rank,
            "rank_change": (rw_rank - rr_rank) if (rw_rank and rr_rank) else None,
            "rewrite_files": rw.get("top5_retrieved_files", [])[:5],
            "rerank_files": rr.get("top5_retrieved_files", [])[:5],
            "change": change,
        })

    return comparison


def _find_rank(per_q_detail: dict, expected_files: set) -> int | None:
    """找到期望文档在 top-5 中的最高排名（1-based），未找到返回 None。"""
    retrieved = per_q_detail.get("top5_retrieved_files", [])
    for i, f in enumerate(retrieved):
        if f in expected_files:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# 对比汇总表
# ---------------------------------------------------------------------------
def print_comparison(rewrite_result: dict, rerank_result: dict):
    """打印改写 vs 改写+Rerank 的对比表。"""
    print("\n" + "=" * 70)
    print("   阶段 3 增量评估：改写 vs 改写+Rerank")
    print("=" * 70)

    header = f"{'配置':<25} {'Top1':>8} {'Top3':>8} {'Top5':>8} {'表格Top1':>10} {'表格Top3':>10}"
    print(header)
    print("-" * 70)

    for label, result in [("rewrite", rewrite_result), ("rewrite+rerank", rerank_result)]:
        recall = result.get("recall", {})
        top1 = recall.get("top1_recall", 0)
        top3 = recall.get("top3_recall", 0)
        top5 = recall.get("top5_recall", 0)
        table_top1 = recall.get("top1_table_recall", 0)
        table_top3 = recall.get("top3_table_recall", 0)
        print(
            f"{label:<25} {top1:>7.1%} {top3:>7.1%} {top5:>7.1%} "
            f"{table_top1:>9.1%} {table_top3:>9.1%}"
        )

    print("=" * 70)

    # 增量
    rw_recall = rewrite_result.get("recall", {})
    rr_recall = rerank_result.get("recall", {})
    print("\n增量（改写+Rerank - 改写）:")
    for key in ["top1_recall", "top3_recall", "top5_recall",
                "top1_table_recall", "top3_table_recall"]:
        delta = rr_recall.get(key, 0) - rw_recall.get(key, 0)
        sign = "+" if delta >= 0 else ""
        print(f"  {key}: {sign}{delta:.1%}")


def print_rank_changes(comparison: list[dict]):
    """打印 rerank 前后排名变化明细。"""
    improved = [c for c in comparison if c["change"].startswith("improved")]
    regressed = [c for c in comparison if c["change"].startswith("regressed")]

    if improved:
        print(f"\n--- 提升 case（共 {len(improved)} 条）---")
        for c in improved:
            table_mark = "[表格]" if c["is_table_question"] else ""
            print(
                f"  Q{c['id']} {table_mark}: {c['question']}"
            )
            print(
                f"    排名: {c['rewrite_rank']} → {c['rerank_rank']}"
                f"（{c['change']}）"
            )
            print(f"    rerank top-3: {c['rerank_files'][:3]}")

    if regressed:
        print(f"\n--- 退步 case（共 {len(regressed)} 条）---")
        for c in regressed:
            table_mark = "[表格]" if c["is_table_question"] else ""
            print(
                f"  Q{c['id']} {table_mark}: {c['question']}"
            )
            print(
                f"    排名: {c['rewrite_rank']} → {c['rerank_rank']}"
                f"（{c['change']}）"
            )
            print(f"    rerank top-3: {c['rerank_files'][:3]}")

    if not improved and not regressed:
        print("\n--- Rerank 未产生命中变化（全部持平）---")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="阶段 3 增量评估：改写 vs 改写+Rerank")
    parser.add_argument(
        "--dataset", type=str, default="tests/golden_dataset.jsonl",
        help="黄金测试集路径",
    )
    parser.add_argument(
        "--fetch-k", type=int, default=10,
        help="Rerank 入口候选数（粗排取多少条）",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Rerank 出口保留数（精排后取多少条）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("   阶段 3 增量评估：改写 vs 改写+Rerank")
    print(f"   Rerank 配置: fetch_k={args.fetch_k}, top_k={args.top_k}")
    print("=" * 70)

    # 1. 加载测试集
    dataset = load_golden_dataset(args.dataset)
    print(f"\n  加载测试集: {len(dataset)} 条")

    # 2. 加载向量库 + 构建 BM25
    print("\n  加载向量库...")
    vectorstore = load_vector_store()

    print("  构建 BM25 检索器...")
    bm25_retriever = BM25Retriever(top_k=10)

    # 3. 构建阶段 2 最优配置（weighted_05_05 + rewrite）
    print("  构建 HybridRetriever (weighted_05_05)...")
    vec_retriever = ScoredVectorRetriever(vectorstore, k=10)
    hybrid = HybridRetriever(
        vec_retriever, bm25_retriever,
        fusion="weighted", weights=[0.5, 0.5], top_k=10,
    )

    print("  构建 RewriteRetriever...")
    rewriter = CachedLLMRewriter()
    # 让 RewriteRetriever 返回更多候选，供 Rerank 精排
    rewrite_retriever = RewriteRetriever(hybrid, rewriter, top_k=max(args.fetch_k, 10))

    # 4. 构建 RerankRetriever
    print("  构建 RerankRetriever (SiliconFlow API)...")
    reranker = SiliconFlowReranker()
    rerank_retriever = RerankRetriever(
        base_retriever=rewrite_retriever,
        reranker=reranker,
        fetch_k=args.fetch_k,
        top_k=args.top_k,
    )

    # 5. 加载阶段 2 改写评估结果（对照组）
    rewrite_result_path = Path("docs/eval_results/rewrite/result.json")
    if rewrite_result_path.exists():
        print(f"\n  加载已有改写评估结果: {rewrite_result_path}")
        with open(rewrite_result_path, encoding="utf-8") as f:
            rewrite_result = json.load(f)
    else:
        print("\n  未找到改写评估结果，先跑改写评估...")
        rewrite_result_raw = evaluate_recall(
            rewrite_retriever, dataset,
            top_k_list=[1, 3, 5], max_fetch_k=10,
        )
        rewrite_result = {
            "label": "rewrite",
            "dataset_path": args.dataset,
            "recall": rewrite_result_raw["summary"],
            "per_question": rewrite_result_raw["per_question"],
            "total_questions": rewrite_result_raw["total_questions"],
            "table_questions": rewrite_result_raw["table_questions"],
        }

    # 6. 跑改写+Rerank 评估（实验组）
    print("\n  跑改写+Rerank 评估（实验组，每题调一次 Rerank API）...")
    rerank_result_raw = evaluate_recall(
        rerank_retriever, dataset,
        top_k_list=[1, 3, 5], max_fetch_k=args.top_k,
    )
    rerank_result = {
        "label": "rerank",
        "dataset_path": args.dataset,
        "recall": rerank_result_raw["summary"],
        "per_question": rerank_result_raw["per_question"],
        "total_questions": rerank_result_raw["total_questions"],
        "table_questions": rerank_result_raw["table_questions"],
        "rerank_config": {
            "model": "BAAI/bge-reranker-v2-m3",
            "backend": "SiliconFlow API",
            "fetch_k": args.fetch_k,
            "top_k": args.top_k,
        },
    }

    # 7. 落盘
    _save_to_eval_results(rerank_result, "rerank", "eval_results")

    # 8. 逐题对比
    comparison = build_comparison(rewrite_result, rerank_result)

    comparison_path = Path("docs/eval_results/rerank_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n  逐题排名对比已保存: {comparison_path}")

    # 9. 打印对比
    print_comparison(rewrite_result, rerank_result)
    print_rank_changes(comparison)

    # 10. 统计变化分类
    change_counts = {}
    for c in comparison:
        change_counts[c["change"]] = change_counts.get(c["change"], 0) + 1
    print("\n  变化分类统计:")
    for change, count in sorted(change_counts.items()):
        print(f"    {change}: {count} 条")

    print(f"\n  所有结果已落盘: docs/eval_results/rerank/")


if __name__ == "__main__":
    main()
