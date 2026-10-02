"""评估脚本：召回命中率 + RAGAS 指标。

本周（第 2-3 周）起召回评估升级为：
- 单次调用 retriever 取 top-10，再切片为 top1/3/5（避免重复 API 调用）
- 逐题明细输出：每题记录期望/实际召回的 source_file、top1/3/5 命中状态
- 整体 + 表格子集双套指标，便于后续每级优化做逐题对比
"""
import difflib
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


def validate_dataset_source_files(
    dataset: list[dict],
    raw_dir: str = "data/raw",
) -> None:
    """校验数据集 source_files 字段是否都命中实际文件，fail-fast 拦截数据 bug。

    背景：source_files 是手打的，容易出现引号类型（ASCII "" vs 中文 ""）、
    前缀数字、空格等细微差异，导致精确字符串匹配时误判为召回失败。
    本函数在评估前做一次集合比对，不命中就报错并提示最近匹配，
    把数据 bug 挡在评估之前。

    Raises:
        ValueError: 任一 source_files 条目未命中实际文件
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"原始文档目录不存在: {raw_dir}")

    actual_files = {p.name for p in raw_path.iterdir() if p.is_file()}
    if not actual_files:
        raise FileNotFoundError(f"原始文档目录为空: {raw_dir}")

    mismatches = []
    for item in dataset:
        qid = item.get("id")
        for sf in item.get("source_files", []):
            if sf in actual_files:
                continue
            # 未命中，找最近匹配（用于提示）
            candidates = difflib.get_close_matches(sf, actual_files, n=1, cutoff=0.0)
            suggestion = candidates[0] if candidates else "<无候选>"
            mismatches.append(
                f"  Q{qid}: 标注 '{sf}' 不在 {raw_dir} 中\n"
                f"         最近匹配: '{suggestion}'"
            )

    if mismatches:
        raise ValueError(
            f"数据集 source_files 校验失败，共 {len(mismatches)} 处不命中：\n"
            + "\n".join(mismatches)
            + f"\n实际文件集合（共 {len(actual_files)} 个）: {sorted(actual_files)}"
        )

    print(f"✅ 数据集 source_files 校验通过（{len(actual_files)} 个实际文件）")


def evaluate_recall(
    retriever,
    dataset: list[dict],
    top_k_list: list[int] = None,
    max_fetch_k: int = 10,
) -> dict:
    """评估召回命中率 Top1/3/5（整体 + 表格子集）。

    关键优化：对每条题只调用一次 retriever（取 top=max_fetch_k），
    再按 top_k_list 切片统计命中，避免重复 API 调用。

    返回结构：
    {
        "summary": {"top1_recall": ..., "top1_table_recall": ..., ...},
        "per_question": [{"id": ..., "question": ..., ...}, ...],
    }
    """
    if top_k_list is None:
        top_k_list = [1, 3, 5]

    # 聚合统计
    agg = {k: {"hit": 0, "total": 0} for k in top_k_list}
    agg_table = {k: {"hit": 0, "total": 0} for k in top_k_list}

    # 逐题明细
    per_question = []

    for item in dataset:
        qid = item.get("id")
        question = item["question"]
        source_files = set(item.get("source_files", []))
        source_tags = set(item.get("source_tags", []))
        use_file_level = len(source_files) > 0
        is_table = bool(item.get("is_table_question", False))

        # 单次取 top=max_fetch_k
        docs = retriever.invoke(question)
        docs = docs[:max_fetch_k]

        # 预计算每条 doc 的"命中判定键"
        retrieved_files = [d.metadata.get("source_file", "") for d in docs]
        retrieved_tags = [d.metadata.get("business_tag", "") for d in docs]

        # 按 k 切片统计
        hits_by_k = {}
        for k in top_k_list:
            if k > max_fetch_k:
                raise ValueError(f"k={k} 超过 max_fetch_k={max_fetch_k}，请调大 max_fetch_k")
            sub_files = set(retrieved_files[:k])
            sub_tags = set(retrieved_tags[:k])

            if use_file_level:
                is_hit = len(source_files & sub_files) > 0
            else:
                is_hit = len(source_tags & sub_tags) > 0

            hits_by_k[k] = is_hit
            agg[k]["total"] += 1
            if is_hit:
                agg[k]["hit"] += 1

            if is_table:
                agg_table[k]["total"] += 1
                if is_hit:
                    agg_table[k]["hit"] += 1

        # 记录逐题明细（取 top-5 的实际召回文件列表，便于归因）
        detail = {
            "id": qid,
            "question": question,
            "is_table_question": is_table,
            "expected_source_files": sorted(source_files) if use_file_level else sorted(source_tags),
            "match_mode": "source_file" if use_file_level else "source_tag",
            "top5_retrieved_files": retrieved_files[:5],
            "top5_retrieved_tags": retrieved_tags[:5],
            "hit": {str(k): hits_by_k[k] for k in top_k_list},
        }
        per_question.append(detail)

    # 汇总
    summary = {}
    for k in top_k_list:
        summary[f"top{k}_recall"] = agg[k]["hit"] / agg[k]["total"] if agg[k]["total"] > 0 else 0
        if agg_table[k]["total"] > 0:
            summary[f"top{k}_table_recall"] = agg_table[k]["hit"] / agg_table[k]["total"]

    return {
        "summary": summary,
        "per_question": per_question,
        "total_questions": len(dataset),
        "table_questions": sum(1 for item in dataset if item.get("is_table_question")),
    }


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
    label: str = "baseline",
    output_dir: Optional[str] = None,
) -> dict:
    """运行完整评估并生成报告。

    参数：
    - label: 本次评估的标签（如 baseline / hybrid / rewrite / rerank），用于命名输出文件
    - output_dir: 若指定，则同时把 JSON + Markdown 明细输出到 docs/<output_dir>/ 下；
                  否则仅写到 output_path（兼容旧行为）
    """
    dataset = load_golden_dataset(dataset_path)

    print(f"📊 加载测试集: {len(dataset)} 条")

    # 0. 校验数据集 source_files 是否命中实际文件（拦截数据 bug）
    validate_dataset_source_files(dataset, raw_dir="data/raw")

    # 1. 召回评估
    print("\n🔍 评估召回率...")
    recall_result = evaluate_recall(retriever, dataset)

    # 2. RAGAS 评估（可选）
    ragas_report = {}
    if qa_chain:
        print("\n🤖 评估生成质量（RAGAS）...")
        ragas_report = evaluate_with_ragas(qa_chain, dataset, max_samples=20)

    # 3. 汇总报告
    full_report = {
        "recall": recall_result["summary"],
        "per_question": recall_result["per_question"],
        "ragas": ragas_report,
        "total_questions": recall_result["total_questions"],
        "table_questions": recall_result["table_questions"],
        "label": label,
    }

    # 4. 生成 Markdown 报告（兼容旧路径）
    _generate_report(full_report, output_path)

    # 5. 若指定 output_dir，额外输出到 docs/eval_results/<label>/
    if output_dir:
        _save_to_eval_results(full_report, label, output_dir)

    return full_report


def _save_to_eval_results(report: dict, label: str, output_dir: str):
    """把评估结果统一落盘到 docs/<output_dir>/<label>/ 下（JSON + Markdown）。"""
    base = Path("docs") / output_dir / label
    base.mkdir(parents=True, exist_ok=True)

    # JSON：完整结果（含逐题明细），供后续阶段做对比
    json_path = base / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown：可读报告
    md_path = base / "report.md"
    _generate_report(report, str(md_path))

    print(f"📁 评估结果已落盘: {json_path} / {md_path}")


def _generate_report(report: dict, output_path: str):
    """生成 Markdown 格式的评估报告（含逐题明细）。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    label = report.get("label", "evaluation")
    lines = [
        f"# 评估报告 - {label}",
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

    # 逐题明细（仅展示未命中的题，便于归因）
    per_question = report.get("per_question", [])
    if per_question:
        # 找出 top-3 仍未命中的题
        missed_top3 = [p for p in per_question if not p["hit"].get("3", False)]
        # 找出 top-3 命中但 top-1 未命中的题
        missed_top1_only = [p for p in per_question if p["hit"].get("3", False) and not p["hit"].get("1", False)]

        lines.extend([
            "",
            "## 逐题明细",
            "",
            f"### Top-3 仍未命中（共 {len(missed_top3)} 条）",
            "",
        ])
        if not missed_top3:
            lines.append("_无_")
        else:
            for p in missed_top3:
                match_mode = p.get("match_mode", "source_file")
                expected = ", ".join(p.get("expected_source_files", []))
                retrieved = ", ".join(p.get("top5_retrieved_files", [])[:3]) if match_mode == "source_file" else ", ".join(p.get("top5_retrieved_tags", [])[:3])
                lines.append(f"- **Q{p['id']}** {p['question']}")
                lines.append(f"  - 期望: `{expected}`")
                lines.append(f"  - Top-3 召回: `{retrieved}`")

        lines.extend([
            "",
            f"### Top-1 未命中但 Top-3 命中（共 {len(missed_top1_only)} 条）",
            "",
        ])
        if not missed_top1_only:
            lines.append("_无_")
        else:
            for p in missed_top1_only[:10]:  # 最多展示 10 条
                match_mode = p.get("match_mode", "source_file")
                expected = ", ".join(p.get("expected_source_files", []))
                retrieved = ", ".join(p.get("top5_retrieved_files", [])[:3]) if match_mode == "source_file" else ", ".join(p.get("top5_retrieved_tags", [])[:3])
                lines.append(f"- **Q{p['id']}** {p['question']}")
                lines.append(f"  - 期望: `{expected}`")
                lines.append(f"  - Top-3 召回: `{retrieved}`")

    lines.extend([
        "",
        "## 分析",
        "",
        "- 表格类问题的召回率通常低于整体，因为表格结构在切分时容易被破坏",
        "- 后续优化重点：表格完整性保留 + Query 改写 + Rerank",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 报告已生成: {output_path}")
