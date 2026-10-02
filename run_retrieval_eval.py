"""召回评估统一对比入口。

用途：
- 第 2-3 周召回侧三级优化（BM25 多路 / Query 改写 / Rerank）期间，
  用同一个入口跑不同 retriever，输出可比的指标 JSON + Markdown 明细表。
- 固定约定：每路召回 top-10，评估看 top1/3/5；指标含整体 + 表格子集两套。

使用方式：
    python run_retrieval_eval.py --label baseline
    python run_retrieval_eval.py --label hybrid --top_k 10
    python run_retrieval_eval.py --label rewrite --dataset tests/golden_dataset.jsonl

产出物：
- docs/eval_results/<label>/result.json（完整结果 + 逐题明细）
- docs/eval_results/<label>/report.md（可读报告）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import evaluate_recall, load_golden_dataset, _save_to_eval_results, _generate_report
from src.vectorstore import load_vector_store


def build_vector_retriever(top_k: int = 10):
    """构建纯向量检索器（基线）。"""
    vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def run_eval(retriever, label: str, dataset_path: str):
    """对指定 retriever 跑评估，输出到 docs/eval_results/<label>/。"""
    dataset = load_golden_dataset(dataset_path)
    print(f"📊 加载测试集: {len(dataset)} 条 (label={label})")

    # 单次 top-10，按 top1/3/5 切片统计
    result = evaluate_recall(retriever, dataset, top_k_list=[1, 3, 5], max_fetch_k=10)

    # 组装报告
    report = {
        "label": label,
        "dataset_path": dataset_path,
        "recall": result["summary"],
        "per_question": result["per_question"],
        "total_questions": result["total_questions"],
        "table_questions": result["table_questions"],
    }

    # 落盘
    _save_to_eval_results(report, label, "eval_results")

    # 打印摘要
    print("\n=== 召回率摘要 ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v:.2%}")
    print(f"\n📁 结果已落盘: docs/eval_results/{label}/")
    return report


def main():
    parser = argparse.ArgumentParser(description="召回评估统一对比入口")
    parser.add_argument("--label", type=str, default="baseline",
                        help="本次评估的标签（baseline/hybrid/rewrite/rerank）")
    parser.add_argument("--dataset", type=str, default="tests/golden_dataset.jsonl",
                        help="黄金测试集路径")
    parser.add_argument("--top_k", type=int, default=10,
                        help="单次 retriever 取多少条（默认 10）")
    args = parser.parse_args()

    print("=" * 60)
    print(f"   召回评估 - label={args.label}")
    print("=" * 60)

    # 当前默认使用纯向量检索器；后续阶段（BM25/改写/Rerank）替换为对应的 retriever 工厂即可
    retriever = build_vector_retriever(top_k=args.top_k)
    run_eval(retriever, args.label, args.dataset)


if __name__ == "__main__":
    main()
