"""阶段 3 增量评估：hybrid vs hybrid+Rerank vs hybrid+Rerank+文档级去重。

对比实验：
- hybrid（阶段 1 最优配置 weighted_05_05，对照组）
- hybrid + rerank（SiliconFlow Rerank API 精排，实验组）
- hybrid + rerank + dedup（文档级去重，同 source_file 最多保留 N 个 chunk）

注意：改写模块已在阶段 2 验证为净效果为负并下线，
      因此 Rerank 直接挂在 hybrid 上，而非 rewrite 上。

产出物：
- docs/eval_results/hybrid_rerank/{result.json, report.md}
- docs/eval_results/hybrid_rerank_dedup/{result.json, report.md}（仅 --dedup 时）
- docs/eval_results/hybrid_rerank_comparison.json（逐题排名对比）
- 控制台输出对比表 + rerank 前后排名明细

使用方式：
    python run_rerank_eval.py
    python run_rerank_eval.py --dedup
    python run_rerank_eval.py --dedup --max-per-source 1
    python run_rerank_eval.py --dedup --boost          # 联合条件时效补偿对照实验
    python run_rerank_eval.py --dedup --boost --boost-value 0.05
    python run_rerank_eval.py --fetch-k 20 --top-k 5
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
    hybrid_result: dict,
    rerank_result: dict,
) -> list[dict]:
    """生成逐题对比明细：rerank 前排名 → rerank 后排名。"""
    hybrid_per_q = {p["id"]: p for p in hybrid_result.get("per_question", [])}
    rerank_per_q = {p["id"]: p for p in rerank_result.get("per_question", [])}

    comparison = []
    for qid in sorted(hybrid_per_q.keys()):
        hb = hybrid_per_q[qid]
        rr = rerank_per_q.get(qid, {})

        expected = set(hb.get("expected_source_files", []))
        hb_rank = _find_rank(hb, expected)
        rr_rank = _find_rank(rr, expected)

        hb_hit1 = hb.get("hit", {}).get("1", False)
        rr_hit1 = rr.get("hit", {}).get("1", False)
        hb_hit3 = hb.get("hit", {}).get("3", False)
        rr_hit3 = rr.get("hit", {}).get("3", False)

        # 变化分类
        if rr_hit1 and not hb_hit1:
            change = "improved_top1"
        elif rr_hit3 and not hb_hit3:
            change = "improved_top3"
        elif hb_hit1 and not rr_hit1:
            change = "regressed_top1"
        elif hb_hit3 and not rr_hit3:
            change = "regressed_top3"
        else:
            change = "unchanged"

        comparison.append({
            "id": qid,
            "question": hb["question"],
            "is_table_question": hb.get("is_table_question", False),
            "hybrid_top1": hb_hit1,
            "rerank_top1": rr_hit1,
            "hybrid_top3": hb_hit3,
            "rerank_top3": rr_hit3,
            "hybrid_rank": hb_rank,
            "rerank_rank": rr_rank,
            "rank_change": (hb_rank - rr_rank) if (hb_rank and rr_rank) else None,
            "hybrid_files": hb.get("top5_retrieved_files", [])[:5],
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
def print_comparison(hybrid_result: dict, rerank_result: dict):
    """打印 hybrid vs hybrid+Rerank 的对比表。"""
    print("\n" + "=" * 70)
    print("   阶段 3 增量评估：hybrid vs hybrid+Rerank（生产链路）")
    print("=" * 70)

    header = f"{'配置':<25} {'Top1':>8} {'Top3':>8} {'Top5':>8} {'表格Top1':>10} {'表格Top3':>10}"
    print(header)
    print("-" * 70)

    for label, result in [("hybrid", hybrid_result), ("hybrid+rerank", rerank_result)]:
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
    hb_recall = hybrid_result.get("recall", {})
    rr_recall = rerank_result.get("recall", {})
    print("\n增量（hybrid+Rerank - hybrid）:")
    for key in ["top1_recall", "top3_recall", "top5_recall",
                "top1_table_recall", "top3_table_recall"]:
        delta = rr_recall.get(key, 0) - hb_recall.get(key, 0)
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
                f"    排名: {c['hybrid_rank']} → {c['rerank_rank']}"
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
                f"    排名: {c['hybrid_rank']} → {c['rerank_rank']}"
                f"（{c['change']}）"
            )
            print(f"    rerank top-3: {c['rerank_files'][:3]}")

    if not improved and not regressed:
        print("\n--- Rerank 未产生命中变化（全部持平）---")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="阶段 3 增量评估：hybrid vs hybrid+Rerank（生产链路）")
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
    parser.add_argument(
        "--dedup", action="store_true",
        help="启用文档级去重（同一 source_file 最多保留 max-per-source 个 chunk）",
    )
    parser.add_argument(
        "--max-per-source", type=int, default=2,
        help="文档级去重：同一 source_file 最多保留几个 chunk（默认 2，需配合 --dedup）",
    )
    parser.add_argument(
        "--boost", action="store_true",
        help="启用主题感知时效补偿（doc_type=通知 + business_tag 与 query 同主题 + 强时效信号）",
    )
    parser.add_argument(
        "--boost-value", type=float, default=0.08,
        help="时效补偿量级（tie-break 级，默认 0.08，需配合 --boost）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("   阶段 3 增量评估：hybrid vs hybrid+Rerank（+文档级去重）")
    dedup_desc = f"启用（max_per_source={args.max_per_source}）" if args.dedup else "关闭"
    boost_desc = f"启用（boost={args.boost_value}）" if args.boost else "关闭"
    print(f"   Rerank 配置: fetch_k={args.fetch_k}, top_k={args.top_k}, 文档级去重: {dedup_desc}, 时效补偿: {boost_desc}")
    print("=" * 70)

    # 1. 加载测试集
    dataset = load_golden_dataset(args.dataset)
    print(f"\n  加载测试集: {len(dataset)} 条")

    # 2. 加载向量库 + 构建 BM25
    print("\n  加载向量库...")
    vectorstore = load_vector_store()

    print("  构建 BM25 检索器...")
    bm25_retriever = BM25Retriever(top_k=10)

    # 3. 构建生产链路基座：hybrid（weighted_05_05，无改写）
    #    去重开启时扩大 hybrid 输出，确保去重后仍有 top_k 条不同文档
    hybrid_top_k = args.top_k * 4 if args.dedup else 10
    print(f"  构建 HybridRetriever (weighted_05_05, top_k={hybrid_top_k})...")
    vec_retriever = ScoredVectorRetriever(vectorstore, k=hybrid_top_k)
    hybrid = HybridRetriever(
        vec_retriever, bm25_retriever,
        fusion="weighted", weights=[0.5, 0.5], top_k=hybrid_top_k,
    )

    # 4. 构建 RerankRetriever（直接挂在 hybrid 上，跳过已下线的改写模块）
    dedup_label = f"_dedup{args.max_per_source}" if args.dedup else ""
    boost_label = "_boost" if args.boost else ""
    result_label = f"hybrid_rerank{dedup_label}{boost_label}"
    print(f"  构建 RerankRetriever (SiliconFlow API, 去重={'开' if args.dedup else '关'}, 时效补偿={'开' if args.boost else '关'})...")
    reranker = SiliconFlowReranker()
    rerank_retriever = RerankRetriever(
        base_retriever=hybrid,
        reranker=reranker,
        fetch_k=args.fetch_k,
        top_k=args.top_k,
        max_per_source=args.max_per_source if args.dedup else None,
        enable_boost=args.boost,
        boost=args.boost_value,
    )

    # 5. 加载 hybrid 评估结果（对照组）
    hybrid_result_path = Path("docs/eval_results/hybrid/result.json")
    if hybrid_result_path.exists():
        print(f"\n  加载已有 hybrid 评估结果: {hybrid_result_path}")
        with open(hybrid_result_path, encoding="utf-8") as f:
            hybrid_result = json.load(f)
    else:
        print("\n  未找到 hybrid 评估结果，先跑 hybrid 评估...")
        hybrid_result_raw = evaluate_recall(
            hybrid, dataset,
            top_k_list=[1, 3, 5], max_fetch_k=10,
        )
        hybrid_result = {
            "label": "hybrid",
            "dataset_path": args.dataset,
            "recall": hybrid_result_raw["summary"],
            "per_question": hybrid_result_raw["per_question"],
            "total_questions": hybrid_result_raw["total_questions"],
            "table_questions": hybrid_result_raw["table_questions"],
        }

    # 6. 跑 hybrid+Rerank 评估（实验组）
    print(f"\n  跑 {result_label} 评估（每题调一次 Rerank API）...")
    rerank_result_raw = evaluate_recall(
        rerank_retriever, dataset,
        top_k_list=[1, 3, 5], max_fetch_k=args.top_k,
    )
    rerank_result = {
        "label": result_label,
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
            "dedup_enabled": args.dedup,
            "max_per_source": args.max_per_source if args.dedup else None,
            "boost_enabled": args.boost,
            "boost_value": args.boost_value if args.boost else None,
        },
    }

    # 7. 落盘
    _save_to_eval_results(rerank_result, result_label, "eval_results")

    # 8. 逐题对比
    comparison = build_comparison(hybrid_result, rerank_result)

    comparison_path = Path("docs/eval_results/hybrid_rerank_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n  逐题排名对比已保存: {comparison_path}")

    # 9. 打印对比
    print_comparison(hybrid_result, rerank_result)
    print_rank_changes(comparison)

    # 10. 统计变化分类
    change_counts = {}
    for c in comparison:
        change_counts[c["change"]] = change_counts.get(c["change"], 0) + 1
    print("\n  变化分类统计:")
    for change, count in sorted(change_counts.items()):
        print(f"    {change}: {count} 条")

    print(f"\n  所有结果已落盘: docs/eval_results/{result_label}/")


if __name__ == "__main__":
    main()
