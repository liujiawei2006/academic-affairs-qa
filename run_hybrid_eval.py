"""BM25 多路召回融合实验脚本。

跑以下配置并输出对比表 + 典型 case：
- vector: 纯向量检索（基线）
- bm25: 纯 BM25 检索
- rrf_k20: RRF 融合，k=20
- rrf_k60: RRF 融合，k=60
- weighted_03_07: 加权融合，向量:BM25 = 0.3:0.7
- weighted_05_05: 加权融合，向量:BM25 = 0.5:0.5
- weighted_07_03: 加权融合，向量:BM25 = 0.7:0.3

产出物：
- docs/eval_results/<label>/{result.json, report.md}  每组配置独立输出
- 控制台输出对比汇总表 + BM25 修复的典型 case

使用方式：
    python run_hybrid_eval.py
    python run_hybrid_eval.py --dataset tests/golden_dataset.jsonl
    python run_hybrid_eval.py --skip vector bm25  # 只跑部分配置
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.documents import Document

from src.evaluation import evaluate_recall, load_golden_dataset, _save_to_eval_results
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.vectorstore import load_vector_store


# ---------------------------------------------------------------------------
# 带分数的向量检索器封装（加权融合需要分数，RRF 只需排名）
# ---------------------------------------------------------------------------
class ScoredVectorRetriever:
    """包装 Milvus retriever，让返回的 Document 带 score 字段。

    原理：用 similarity_search_with_score() 取余弦距离，
    转为 similarity = 1 - distance 存入 metadata["score"]。
    """

    def __init__(self, vectorstore, k: int = 10):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query: str) -> list[Document]:
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
# 实验配置定义
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # (label, mode, extra_kwargs)
    ("vector", "vector", {}),
    ("bm25", "bm25", {}),
    ("rrf_k20", "rrf", {"k": 20}),
    ("rrf_k60", "rrf", {"k": 60}),
    ("weighted_03_07", "weighted", {"weights": [0.3, 0.7]}),
    ("weighted_05_05", "weighted", {"weights": [0.5, 0.5]}),
    ("weighted_07_03", "weighted", {"weights": [0.7, 0.3]}),
]


def build_retriever(mode: str, vectorstore, bm25_retriever, **kwargs):
    """根据模式构建对应的 retriever。"""
    if mode == "vector":
        return vectorstore.as_retriever(search_kwargs={"k": 10})
    elif mode == "bm25":
        return bm25_retriever
    elif mode == "rrf":
        vec_retriever = ScoredVectorRetriever(vectorstore, k=10)
        return HybridRetriever(
            vec_retriever, bm25_retriever,
            fusion="rrf", k=kwargs.get("k", 60), top_k=10,
        )
    elif mode == "weighted":
        vec_retriever = ScoredVectorRetriever(vectorstore, k=10)
        return HybridRetriever(
            vec_retriever, bm25_retriever,
            fusion="weighted", weights=kwargs.get("weights"), top_k=10,
        )
    else:
        raise ValueError(f"未知模式: {mode}")


# ---------------------------------------------------------------------------
# 对比汇总表 + 典型 case 提取
# ---------------------------------------------------------------------------
def print_comparison_table(all_results: dict[str, dict]):
    """打印各配置的召回率对比表。"""
    print("\n" + "=" * 80)
    print("   BM25 多路召回融合实验 —— 对比汇总")
    print("=" * 80)

    # 表头
    header = f"{'配置':<20} {'Top1':>8} {'Top3':>8} {'Top5':>8} {'表格Top1':>10} {'表格Top3':>10}"
    print(header)
    print("-" * 80)

    for label, result in all_results.items():
        recall = result["recall"]
        top1 = recall.get("top1_recall", 0)
        top3 = recall.get("top3_recall", 0)
        top5 = recall.get("top5_recall", 0)
        table_top1 = recall.get("top1_table_recall", 0)
        table_top3 = recall.get("top3_table_recall", 0)
        print(
            f"{label:<20} {top1:>7.1%} {top3:>7.1%} {top5:>7.1%} "
            f"{table_top1:>9.1%} {table_top3:>9.1%}"
        )

    print("=" * 80)


def find_bm25_fix_cases(all_results: dict[str, dict]) -> list[dict]:
    """找出 BM25 修复的典型 case：纯向量 miss → 融合后 hit。"""
    vector_result = all_results.get("vector", {})
    best_hybrid_result = None
    best_label = ""

    # 找最好的融合配置（按 Top3 排序）
    for label, result in all_results.items():
        if label in ("vector", "bm25"):
            continue
        top3 = result["recall"].get("top3_recall", 0)
        if best_hybrid_result is None or top3 > best_hybrid_result["recall"].get("top3_recall", 0):
            best_hybrid_result = result
            best_label = label

    if not best_hybrid_result:
        return []

    vec_per_q = {p["id"]: p for p in vector_result.get("per_question", [])}
    hybrid_per_q = {p["id"]: p for p in best_hybrid_result.get("per_question", [])}

    fix_cases = []
    for qid in vec_per_q:
        vec_hit = vec_per_q[qid].get("hit", {}).get("3", False)
        hybrid_hit = hybrid_per_q.get(qid, {}).get("hit", {}).get("3", False)

        # 向量 top-3 miss → 融合后 hit
        if not vec_hit and hybrid_hit:
            fix_cases.append({
                "id": qid,
                "question": vec_per_q[qid]["question"],
                "expected": vec_per_q[qid].get("expected_source_files", []),
                "vector_top5": vec_per_q[qid].get("top5_retrieved_files", [])[:3],
                "hybrid_top5": hybrid_per_q[qid].get("top5_retrieved_files", [])[:3],
                "hybrid_label": best_label,
            })

    return fix_cases


def print_fix_cases(fix_cases: list[dict]):
    """打印 BM25 修复的典型 case。"""
    if not fix_cases:
        print("\n未找到 BM25 修复的典型 case（向量 top-3 全部命中或融合后仍 miss）")
        return

    print(f"\n{'=' * 80}")
    print(f"   BM25 修复的典型 case（共 {len(fix_cases)} 条）")
    print(f"{'=' * 80}")

    for case in fix_cases[:10]:  # 最多展示 10 条
        print(f"\n  Q{case['id']}: {case['question']}")
        print(f"    期望来源: {case['expected']}")
        print(f"    纯向量 top-3: {case['vector_top5']}")
        print(f"    融合后 top-3 ({case['hybrid_label']}): {case['hybrid_top5']}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BM25 多路召回融合实验")
    parser.add_argument(
        "--dataset", type=str, default="tests/golden_dataset.jsonl",
        help="黄金测试集路径",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="跳过某些配置（如 --skip vector bm25）",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("   BM25 多路召回融合实验")
    print("=" * 80)

    # 1. 加载测试集
    dataset = load_golden_dataset(args.dataset)
    print(f"\n📊 加载测试集: {len(dataset)} 条")

    # 2. 加载向量库（全局共享，只加载一次）
    print("\n📦 加载向量库...")
    vectorstore = load_vector_store()

    # 3. 构建 BM25 检索器（全局共享）
    print("\n🔧 构建 BM25 检索器...")
    bm25_retriever = BM25Retriever(top_k=10)

    # 4. 逐配置跑评估
    all_results = {}
    for label, mode, extra_kwargs in EXPERIMENTS:
        if label in args.skip:
            print(f"\n⏭️  跳过配置: {label}")
            continue

        print(f"\n{'─' * 60}")
        print(f"🔬 运行配置: {label} (mode={mode}, kwargs={extra_kwargs})")
        print(f"{'─' * 60}")

        retriever = build_retriever(mode, vectorstore, bm25_retriever, **extra_kwargs)
        result = evaluate_recall(retriever, dataset, top_k_list=[1, 3, 5], max_fetch_k=10)

        report = {
            "label": label,
            "dataset_path": args.dataset,
            "mode": mode,
            "config": extra_kwargs,
            "recall": result["summary"],
            "per_question": result["per_question"],
            "total_questions": result["total_questions"],
            "table_questions": result["table_questions"],
        }

        _save_to_eval_results(report, label, "eval_results")
        all_results[label] = report

        # 打印当前配置的摘要
        print(f"  Top1: {result['summary'].get('top1_recall', 0):.1%}  "
              f"Top3: {result['summary'].get('top3_recall', 0):.1%}  "
              f"Top5: {result['summary'].get('top5_recall', 0):.1%}")

    # 5. 输出对比汇总表
    print_comparison_table(all_results)

    # 6. 提取 BM25 修复的典型 case
    fix_cases = find_bm25_fix_cases(all_results)
    print_fix_cases(fix_cases)

    # 7. 保存典型 case 到文件
    if fix_cases:
        cases_path = Path("docs/eval_results/bm25_fix_cases.json")
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cases_path, "w", encoding="utf-8") as f:
            json.dump(fix_cases, f, ensure_ascii=False, indent=2)
        print(f"\n📁 BM25 修复 case 已保存: {cases_path}")

    print(f"\n📁 所有评估结果已落盘: docs/eval_results/<label>/")


if __name__ == "__main__":
    main()
