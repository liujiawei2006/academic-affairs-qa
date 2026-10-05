"""阶段 2 增量评估：融合 vs 融合+改写。

对比实验：
- hybrid（阶段 1 最优配置 weighted_05_05）
- hybrid + rewrite（术语词典 + LLM 多查询改写）

产出物：
- docs/eval_results/<label>/{result.json, report.md}
- docs/eval_results/rewrite_comparison.json（逐题对比明细 + 改写样例表）
- 控制台输出对比表 + 改写样例

使用方式：
    python run_rewrite_eval.py
    python run_rewrite_eval.py --dataset tests/golden_dataset.jsonl
    python run_rewrite_eval.py --clear-cache   # 清空改写缓存后重跑
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import evaluate_recall, load_golden_dataset, _save_to_eval_results
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.vectorstore import load_vector_store
from src.query.rewriter import CachedLLMRewriter, RewriteRetriever


# ---------------------------------------------------------------------------
# 带分数的向量检索器封装（加权融合需要分数）
# ---------------------------------------------------------------------------
class ScoredVectorRetriever:
    """包装 Milvus retriever，让返回的 Document 带 score 字段。

    原理：用 similarity_search_with_score() 取余弦距离，
    转为 similarity = 1 - distance 存入 metadata["score"]。
    """

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
# 逐题对比
# ---------------------------------------------------------------------------
def build_comparison(
    hybrid_result: dict,
    rewrite_result: dict,
    rewriter: CachedLLMRewriter,
) -> list[dict]:
    """生成逐题对比明细：原问题 / 改写结果 / 命中变化。"""
    hybrid_per_q = {p["id"]: p for p in hybrid_result.get("per_question", [])}
    rewrite_per_q = {p["id"]: p for p in rewrite_result.get("per_question", [])}

    comparison = []
    for qid in sorted(hybrid_per_q.keys()):
        h = hybrid_per_q[qid]
        r = rewrite_per_q.get(qid, {})

        # 获取改写结果（用于样例表展示）
        llm_variants = rewriter.rewrite(h["question"])

        # 判断命中变化
        h_hit1 = h.get("hit", {}).get("1", False)
        r_hit1 = r.get("hit", {}).get("1", False)
        h_hit3 = h.get("hit", {}).get("3", False)
        r_hit3 = r.get("hit", {}).get("3", False)

        # 排名变化：期望文档在 top-5 中的位置
        expected = set(h.get("expected_source_files", []))
        h_rank = _find_rank(h, expected)
        r_rank = _find_rank(r, expected)

        # 变化分类
        if r_hit1 and not h_hit1:
            change = "improved_top1"
        elif r_hit3 and not h_hit3:
            change = "improved_top3"
        elif h_hit1 and not r_hit1:
            change = "regressed_top1"
        elif h_hit3 and not r_hit3:
            change = "regressed_top3"
        else:
            change = "unchanged"

        comparison.append({
            "id": qid,
            "question": h["question"],
            "llm_variants": llm_variants,
            "hybrid_top1": h_hit1,
            "rewrite_top1": r_hit1,
            "hybrid_top3": h_hit3,
            "rewrite_top3": r_hit3,
            "hybrid_rank": h_rank,
            "rewrite_rank": r_rank,
            "rank_change": (h_rank - r_rank) if (h_rank and r_rank) else None,
            "hybrid_files": h.get("top5_retrieved_files", [])[:3],
            "rewrite_files": r.get("top5_retrieved_files", [])[:3],
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
def print_comparison(hybrid_result: dict, rewrite_result: dict):
    """打印融合 vs 融合+改写的对比表。"""
    print("\n" + "=" * 70)
    print("   阶段 2 增量评估：融合 vs 融合+改写")
    print("=" * 70)

    header = f"{'配置':<25} {'Top1':>8} {'Top3':>8} {'Top5':>8} {'表格Top1':>10} {'表格Top3':>10}"
    print(header)
    print("-" * 70)

    for label, result in [("hybrid", hybrid_result), ("hybrid+rewrite", rewrite_result)]:
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
    h_recall = hybrid_result.get("recall", {})
    r_recall = rewrite_result.get("recall", {})
    print("\n增量（融合+改写 - 融合）:")
    for key in ["top1_recall", "top3_recall", "top5_recall"]:
        delta = r_recall.get(key, 0) - h_recall.get(key, 0)
        sign = "+" if delta >= 0 else ""
        print(f"  {key}: {sign}{delta:.1%}")


def print_rewrite_samples(comparison: list[dict]):
    """打印改写样例表（按变化分类展示）。"""
    improved = [c for c in comparison if c["change"].startswith("improved")]
    regressed = [c for c in comparison if c["change"].startswith("regressed")]

    if improved:
        print(f"\n--- 提升 case（共 {len(improved)} 条，展示前 5 条）---")
        for c in improved[:5]:
            print(f"  Q{c['id']}: {c['question']}")
            if c['llm_variants']:
                for v in c['llm_variants']:
                    print(f"    LLM 变体: {v}")
            print(f"    hybrid top-3: {c['hybrid_files']}")
            print(f"    rewrite top-3: {c['rewrite_files']}")
            print()

    if regressed:
        print(f"\n--- 退步 case（共 {len(regressed)} 条）---")
        for c in regressed:
            print(f"  Q{c['id']}: {c['question']}")
            if c['llm_variants']:
                for v in c['llm_variants']:
                    print(f"    LLM 变体: {v}")
            print(f"    hybrid top-3: {c['hybrid_files']}")
            print(f"    rewrite top-3: {c['rewrite_files']}")
            print()

    if not improved and not regressed:
        print("\n--- 改写未产生命中变化（全部持平）---")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="阶段 2 增量评估：融合 vs 融合+改写")
    parser.add_argument(
        "--dataset", type=str, default="tests/golden_dataset.jsonl",
        help="黄金测试集路径",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="清空改写缓存后重跑（修改 prompt 后需要）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("   阶段 2 增量评估：融合 vs 融合+改写")
    print("=" * 70)

    # 1. 加载测试集
    dataset = load_golden_dataset(args.dataset)
    print(f"\n  加载测试集: {len(dataset)} 条")

    # 2. 加载向量库 + 构建 BM25
    print("\n  加载向量库...")
    vectorstore = load_vector_store()

    print("  构建 BM25 检索器...")
    bm25_retriever = BM25Retriever(top_k=10)

    # 3. 构建阶段 1 最优配置（weighted_05_05）
    print("  构建 HybridRetriever (weighted_05_05)...")
    vec_retriever = ScoredVectorRetriever(vectorstore, k=10)
    hybrid = HybridRetriever(
        vec_retriever, bm25_retriever,
        fusion="weighted", weights=[0.5, 0.5], top_k=10,
    )

    # 4. 构建 RewriteRetriever
    print("  构建 RewriteRetriever...")
    rewriter = CachedLLMRewriter()
    if args.clear_cache:
        rewriter.clear_cache()
    rewrite_retriever = RewriteRetriever(hybrid, rewriter)

    # 5. 跑融合评估（对照组）
    print("\n  [1/2] 跑融合评估（对照组）...")
    hybrid_result = evaluate_recall(hybrid, dataset, top_k_list=[1, 3, 5], max_fetch_k=10)

    # 6. 跑融合+改写评估（实验组）
    print("\n  [2/2] 跑融合+改写评估（实验组）...")
    rewrite_result = evaluate_recall(rewrite_retriever, dataset, top_k_list=[1, 3, 5], max_fetch_k=10)

    # 7. 组装报告并落盘
    hybrid_report = {
        "label": "hybrid",
        "dataset_path": args.dataset,
        "recall": hybrid_result["summary"],
        "per_question": hybrid_result["per_question"],
        "total_questions": hybrid_result["total_questions"],
        "table_questions": hybrid_result["table_questions"],
    }
    rewrite_report = {
        "label": "rewrite",
        "dataset_path": args.dataset,
        "recall": rewrite_result["summary"],
        "per_question": rewrite_result["per_question"],
        "total_questions": rewrite_result["total_questions"],
        "table_questions": rewrite_result["table_questions"],
    }

    _save_to_eval_results(hybrid_report, "hybrid", "eval_results")
    _save_to_eval_results(rewrite_report, "rewrite", "eval_results")

    # 8. 逐题对比
    comparison = build_comparison(hybrid_report, rewrite_report, rewriter)

    # 9. 保存改写样例表
    comparison_path = Path("docs/eval_results/rewrite_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n  改写样例表已保存: {comparison_path}")

    # 10. 打印对比
    print_comparison(hybrid_report, rewrite_report)
    print_rewrite_samples(comparison)

    # 11. 统计变化分类
    change_counts = {}
    for c in comparison:
        change_counts[c["change"]] = change_counts.get(c["change"], 0) + 1
    print("\n  变化分类统计:")
    for change, count in sorted(change_counts.items()):
        print(f"    {change}: {count} 条")

    print(f"\n  所有结果已落盘: docs/eval_results/hybrid/ 和 docs/eval_results/rewrite/")


if __name__ == "__main__":
    main()
