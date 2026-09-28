"""评估脚本：召回命中率 + RAGAS 指标。"""
import json
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document


def load_golden_dataset(path: str = "tests/golden_dataset.jsonl") -> list[dict]:
    """加载黄金测试集。"""
    dataset = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def evaluate_recall(
    retriever,
    dataset: list[dict],
    top_k_list: list[int] = None,
) -> dict:
    """评估召回命中率 Top1/3/5。"""
    if top_k_list is None:
        top_k_list = [1, 3, 5]

    results = {k: {"hit": 0, "total": 0} for k in top_k_list}
    table_results = {k: {"hit": 0, "total": 0} for k in top_k_list}

    for item in dataset:
        question = item["question"]
        source_tags = set(item.get("source_tags", []))

        for k in top_k_list:
            docs = retriever.invoke(question)
            retrieved = docs[:k]

            # 检查召回的文档是否包含相关标签
            retrieved_tags = set()
            for doc in retrieved:
                tag = doc.metadata.get("business_tag", "")
                if tag:
                    retrieved_tags.add(tag)

            # 判断是否命中（有交集）
            is_hit = len(source_tags & retrieved_tags) > 0

            results[k]["total"] += 1
            if is_hit:
                results[k]["hit"] += 1

            # 表格类问题单独统计
            if item.get("is_table_question", False):
                table_results[k]["total"] += 1
                if is_hit:
                    table_results[k]["hit"] += 1

    # 计算命中率
    report = {}
    for k in top_k_list:
        report[f"top{k}_recall"] = results[k]["hit"] / results[k]["total"] if results[k]["total"] > 0 else 0
        if table_results[k]["total"] > 0:
            report[f"top{k}_table_recall"] = table_results[k]["hit"] / table_results[k]["total"]

    return report


def evaluate_with_ragas(
    qa_chain,
    dataset: list[dict],
    max_samples: Optional[int] = None,
) -> dict:
    """使用 RAGAS 评估生成质量（相关性、事实一致性）。"""
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from datasets import Dataset
    except ImportError:
        print("⚠️  RAGAS 未安装，跳过该评估")
        return {}

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    samples = dataset[:max_samples] if max_samples else dataset

    for item in samples:
        question = item["question"]
        ground_truth = item["answer"]

        # 获取 RAG 回答和上下文
        try:
            result = qa_chain.invoke(question)
            answer = result if isinstance(result, str) else result.get("answer", str(result))

            # 获取检索的上下文
            if hasattr(qa_chain, "retriever"):
                docs = qa_chain.retriever.invoke(question)
                contexts = [doc.page_content for doc in docs]
            else:
                contexts = [""]

            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(ground_truth)
        except Exception as e:
            print(f"⚠️  评估问题 '{question}' 失败: {e}")

    if not eval_data["question"]:
        return {}

    # 转换为 HuggingFace Dataset
    hf_dataset = Dataset.from_dict(eval_data)

    # 运行评估
    metrics = [answer_relevancy, faithfulness]
    eval_result = evaluate(hf_dataset, metrics=metrics)

    return {
        "answer_relevancy": eval_result["answer_relevancy"],
        "faithfulness": eval_result["faithfulness"],
    }


def run_full_evaluation(
    retriever,
    qa_chain=None,
    dataset_path: str = "tests/golden_dataset.jsonl",
    output_path: str = "docs/baseline_report.md",
) -> dict:
    """运行完整评估并生成报告。"""
    dataset = load_golden_dataset(dataset_path)

    print(f"📊 加载测试集: {len(dataset)} 条")

    # 1. 召回评估
    print("\n🔍 评估召回率...")
    recall_report = evaluate_recall(retriever, dataset)

    # 2. RAGAS 评估（可选）
    ragas_report = {}
    if qa_chain:
        print("\n🤖 评估生成质量（RAGAS）...")
        ragas_report = evaluate_with_ragas(qa_chain, dataset, max_samples=20)

    # 3. 汇总报告
    full_report = {
        "recall": recall_report,
        "ragas": ragas_report,
        "total_questions": len(dataset),
        "table_questions": sum(1 for item in dataset if item.get("is_table_question")),
    }

    # 4. 生成 Markdown 报告
    _generate_report(full_report, output_path)

    return full_report


def _generate_report(report: dict, output_path: str):
    """生成 Markdown 格式的评估报告。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 基线评估报告",
        "",
        f"**评估时间**: 自动生成",
        f"**测试集规模**: {report['total_questions']} 条（表格类 {report['table_questions']} 条）",
        "",
        "## 召回率",
        "",
    ]

    recall = report.get("recall", {})
    for key, value in recall.items():
        lines.append(f"- **{key}**: {value:.2%}")

    if report.get("ragas"):
        lines.extend([
            "",
            "## 生成质量（RAGAS）",
            "",
        ])
        for key, value in report["ragas"].items():
            lines.append(f"- **{key}**: {value:.3f}")

    lines.extend([
        "",
        "## 分析",
        "",
        "- 表格类问题的召回率通常低于整体，因为表格结构在切分时容易被破坏",
        "- 后续优化重点：表格完整性保留 + Query 改写 + Rerank",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📄 报告已生成: {output_path}")
