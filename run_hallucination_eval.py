"""E2 幻觉率评估脚本：对比优化前（朴素 prompt）vs 优化后（约束 prompt + 校验 + 兜底）。

评估方法：规则匹配（基于 golden_dataset_answer.jsonl 的 expected_values / expected_conditions / expected_answer_points）
评估指标：
  - 数值准确率：expected_values 在答案中的命中率
  - 条件覆盖率：expected_conditions 在答案中的覆盖率
  - 负样本拒答率：负样本是否正确触发兜底
  - 综合幻觉率：(缺失值 + 缺失条件 + 编造值) / 总检查项

用法：
    python run_hallucination_eval.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

from src.generation.answerer import Answerer
from src.generation.llm_client import LLMClient
from src.generation.fallback import FALLBACK_MESSAGE, pre_llm_fallback_check
from src.generation.verifier import verify
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import RerankRetriever, SiliconFlowReranker
from src.vectorstore import load_vector_store


# ---------------------------------------------------------------------------
# 朴素 prompt（优化前基线）
# ---------------------------------------------------------------------------

NAIVE_PROMPT_TEMPLATE = """\
你是高校教务政策知识库问答助手。请根据以下参考资料回答学生的问题。

参考资料：
{context}

学生问题：{question}

回答："""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_answer_dataset(path: str = "tests/golden_dataset_answer.jsonl") -> list[dict]:
    """加载答案级评估集。"""
    dataset = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def build_retriever():
    """构建生产检索链路。"""
    vec_store = load_vector_store()
    vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever()
    hybrid = HybridRetriever(vec_retriever, bm25_retriever, fusion="rrf", k=60)
    reranker = SiliconFlowReranker()
    return RerankRetriever(hybrid, reranker, fetch_k=10, top_k=5, max_per_source=2)


def normalize(s: str) -> str:
    """归一化字符串，便于比对。"""
    s = re.sub(r"\s+", "", s)
    s = s.translate(str.maketrans("０１２３４５６７８９．％，。；：", "0123456789.%.,;:"))
    return s


def check_value_in_answer(value: str, answer: str) -> bool:
    """检查期望数值是否出现在答案中（归一化后匹配）。"""
    return normalize(value) in normalize(answer)


def check_condition_in_answer(condition: str, answer: str) -> bool:
    """检查期望条件是否在答案中被提及（核心关键词匹配）。"""
    # 提取核心关键词（去除标点、数字，保留中文）
    core = re.sub(r"[^\u4e00-\u9fff]", "", condition)
    if len(core) < 2:
        return True  # 太短的条件跳过（如"无"）
    # 核心关键词至少 60% 出现在答案中
    core_chars = list(core)
    matched = sum(1 for c in core_chars if c in answer)
    return matched / len(core_chars) >= 0.6


def run_naive_answer(question: str, docs: list[Document], llm: LLMClient) -> str:
    """用朴素 prompt 生成答案（模拟优化前）。"""
    if not docs:
        return "（无相关参考资料）"
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[{i}] {doc.page_content}")
    context = "\n\n".join(context_parts)
    prompt = NAIVE_PROMPT_TEMPLATE.format(context=context, question=question)
    return llm.invoke(prompt)


# ---------------------------------------------------------------------------
# 单题评估
# ---------------------------------------------------------------------------

def evaluate_single(
    item: dict,
    retriever,
    answerer: Answerer,
    llm: LLMClient,
) -> dict:
    """评估单道题的幻觉情况。"""
    qid = item["id"]
    question = item["question"]
    is_negative = item.get("negative_sample", False)
    is_table = item.get("is_table_question", False)
    expected_values = item.get("expected_values", [])
    expected_conditions = item.get("expected_conditions", [])
    expected_points = item.get("expected_answer_points", [])

    # 检索
    docs = retriever.invoke(question)[:5]

    result = {
        "id": qid,
        "question": question,
        "is_negative": is_negative,
        "is_table": is_table,
    }

    # ---- 负样本 ----
    if is_negative:
        # 优化后：用 Answerer（含兜底）
        opt_result = answerer.answer(question, enable_verify=False)
        opt_answer = opt_result["answer"]
        opt_fallback = opt_result["fallback"].should_fallback

        # 优化前：用朴素 prompt（无兜底）
        pre_fb = pre_llm_fallback_check(question, docs)
        if pre_fb.should_fallback:
            naive_answer = FALLBACK_MESSAGE
            naive_fallback = True
        else:
            naive_answer = run_naive_answer(question, docs, llm)
            naive_fallback = False

        result.update({
            "naive_answer": naive_answer,
            "optimized_answer": opt_answer,
            "naive_correctly_rejected": naive_fallback,
            "optimized_correctly_rejected": opt_fallback,
        })
        return result

    # ---- 正样本 ----
    # 优化前：朴素 prompt
    pre_fb = pre_llm_fallback_check(question, docs)
    if pre_fb.should_fallback:
        naive_answer = FALLBACK_MESSAGE
    else:
        naive_answer = run_naive_answer(question, docs, llm)

    # 优化后：约束 prompt + 校验 + 兜底
    opt_result = answerer.answer(question)
    opt_answer = opt_result["answer"]
    verification = opt_result.get("verification")

    # 评估数值准确率
    naive_value_hits = sum(1 for v in expected_values if check_value_in_answer(v, naive_answer))
    opt_value_hits = sum(1 for v in expected_values if check_value_in_answer(v, opt_answer))
    total_values = len(expected_values)

    # 评估条件覆盖率
    naive_cond_hits = sum(1 for c in expected_conditions if check_condition_in_answer(c, naive_answer))
    opt_cond_hits = sum(1 for c in expected_conditions if check_condition_in_answer(c, opt_answer))
    total_conditions = len(expected_conditions)

    # C3 校验幻觉数
    opt_hallucination_count = 0
    if verification:
        opt_hallucination_count = verification.hallucination_count

    result.update({
        "naive_answer": naive_answer,
        "optimized_answer": opt_answer,
        "expected_values": expected_values,
        "expected_conditions": expected_conditions,
        "naive_value_hits": naive_value_hits,
        "optimized_value_hits": opt_value_hits,
        "total_values": total_values,
        "naive_condition_hits": naive_cond_hits,
        "optimized_condition_hits": opt_cond_hits,
        "total_conditions": total_conditions,
        "optimized_hallucination_count": opt_hallucination_count,
    })
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("E2 幻觉率评估：优化前 vs 优化后")
    print("=" * 60)

    # 加载数据
    dataset = load_answer_dataset()
    print(f"加载评估集: {len(dataset)} 条")

    positive = [d for d in dataset if not d.get("negative_sample")]
    negative = [d for d in dataset if d.get("negative_sample")]
    print(f"  正样本: {len(positive)}, 负样本: {len(negative)}")

    # 初始化
    print("\n加载检索链路...")
    retriever = build_retriever()
    llm = LLMClient()
    answerer = Answerer(retriever, llm)

    # 逐题评估
    results = []
    for i, item in enumerate(dataset):
        qid = item["id"]
        print(f"\r评估中... [{i+1}/{len(dataset)}] Q{qid}", end="", flush=True)
        r = evaluate_single(item, retriever, answerer, llm)
        results.append(r)
        time.sleep(0.3)  # 限流

    print("\n")

    # ---- 汇总统计 ----
    pos_results = [r for r in results if not r["is_negative"]]
    neg_results = [r for r in results if r["is_negative"]]

    # 正样本指标
    total_values = sum(r["total_values"] for r in pos_results)
    total_conditions = sum(r["total_conditions"] for r in pos_results)

    naive_value_total = sum(r["naive_value_hits"] for r in pos_results)
    opt_value_total = sum(r["optimized_value_hits"] for r in pos_results)
    naive_cond_total = sum(r["naive_condition_hits"] for r in pos_results)
    opt_cond_total = sum(r["optimized_condition_hits"] for r in pos_results)
    opt_hallucination_total = sum(r["optimized_hallucination_count"] for r in pos_results)

    naive_value_rate = naive_value_total / total_values if total_values > 0 else 0
    opt_value_rate = opt_value_total / total_values if total_values > 0 else 0
    naive_cond_rate = naive_cond_total / total_conditions if total_conditions > 0 else 0
    opt_cond_rate = opt_cond_total / total_conditions if total_conditions > 0 else 0

    # 综合幻觉率（优化后）
    total_checks = total_values + total_conditions
    total_issues = (total_values - opt_value_total) + (total_conditions - opt_cond_total) + opt_hallucination_total
    opt_hallucination_rate = total_issues / total_checks if total_checks > 0 else 0

    # 综合幻觉率（优化前）
    naive_total_issues = (total_values - naive_value_total) + (total_conditions - naive_cond_total)
    naive_hallucination_rate = naive_total_issues / total_checks if total_checks > 0 else 0

    # 负样本拒答率
    naive_reject = sum(1 for r in neg_results if r["naive_correctly_rejected"])
    opt_reject = sum(1 for r in neg_results if r["optimized_correctly_rejected"])
    naive_reject_rate = naive_reject / len(neg_results) if neg_results else 0
    opt_reject_rate = opt_reject / len(neg_results) if neg_results else 0

    # 表格子集
    table_results = [r for r in pos_results if r["is_table"]]
    table_total_values = sum(r["total_values"] for r in table_results)
    table_naive_val = sum(r["naive_value_hits"] for r in table_results)
    table_opt_val = sum(r["optimized_value_hits"] for r in table_results)
    table_naive_rate = table_naive_val / table_total_values if table_total_values > 0 else 0
    table_opt_rate = table_opt_val / table_total_values if table_total_values > 0 else 0

    # ---- 输出报告 ----
    summary = {
        "positive_count": len(pos_results),
        "negative_count": len(neg_results),
        "naive": {
            "value_accuracy": round(naive_value_rate, 4),
            "condition_coverage": round(naive_cond_rate, 4),
            "hallucination_rate": round(naive_hallucination_rate, 4),
            "negative_reject_rate": round(naive_reject_rate, 4),
            "table_value_accuracy": round(table_naive_rate, 4),
        },
        "optimized": {
            "value_accuracy": round(opt_value_rate, 4),
            "condition_coverage": round(opt_cond_rate, 4),
            "hallucination_rate": round(opt_hallucination_rate, 4),
            "negative_reject_rate": round(opt_reject_rate, 4),
            "table_value_accuracy": round(table_opt_rate, 4),
        },
        "delta": {
            "value_accuracy": round(opt_value_rate - naive_value_rate, 4),
            "condition_coverage": round(opt_cond_rate - naive_cond_rate, 4),
            "hallucination_rate": round(naive_hallucination_rate - opt_hallucination_rate, 4),
            "negative_reject_rate": round(opt_reject_rate - naive_reject_rate, 4),
        },
    }

    # 打印摘要
    print("=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"{'指标':<20} {'优化前':>10} {'优化后':>10} {'变化':>10}")
    print("-" * 50)
    print(f"{'数值准确率':<20} {naive_value_rate:>10.1%} {opt_value_rate:>10.1%} {opt_value_rate-naive_value_rate:>+10.1%}")
    print(f"{'条件覆盖率':<20} {naive_cond_rate:>10.1%} {opt_cond_rate:>10.1%} {opt_cond_rate-naive_cond_rate:>+10.1%}")
    print(f"{'综合幻觉率':<20} {naive_hallucination_rate:>10.1%} {opt_hallucination_rate:>10.1%} {naive_hallucination_rate-opt_hallucination_rate:>+10.1%}")
    print(f"{'负样本拒答率':<20} {naive_reject_rate:>10.1%} {opt_reject_rate:>10.1%} {opt_reject_rate-naive_reject_rate:>+10.1%}")
    print(f"{'表格数值准确率':<16} {table_naive_rate:>10.1%} {table_opt_rate:>10.1%} {table_opt_rate-table_naive_rate:>+10.1%}")
    print("=" * 60)

    # 落盘
    output_dir = Path("docs/eval_results/hallucination")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    output_data = {
        "summary": summary,
        "per_question": results,
    }
    json_path = output_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Markdown
    md_path = output_dir / "report.md"
    _generate_report(summary, pos_results, neg_results, str(md_path))

    print(f"\n结果已落盘: {json_path} / {md_path}")

    return output_data


def _generate_report(summary: dict, pos_results: list, neg_results: list, output_path: str):
    """生成 Markdown 报告。"""
    naive = summary["naive"]
    opt = summary["optimized"]
    delta = summary["delta"]

    lines = [
        "# 幻觉率评估报告",
        "",
        f"**评估集**: {summary['positive_count']} 正样本 + {summary['negative_count']} 负样本",
        "",
        "## 对比总览",
        "",
        "| 指标 | 优化前（朴素 prompt） | 优化后（约束 prompt + 校验 + 兜底） | 变化 |",
        "|------|----------------------|--------------------------------------|------|",
        f"| 数值准确率 | {naive['value_accuracy']:.1%} | {opt['value_accuracy']:.1%} | {delta['value_accuracy']:+.1%} |",
        f"| 条件覆盖率 | {naive['condition_coverage']:.1%} | {opt['condition_coverage']:.1%} | {delta['condition_coverage']:+.1%} |",
        f"| 综合幻觉率 | {naive['hallucination_rate']:.1%} | {opt['hallucination_rate']:.1%} | {delta['hallucination_rate']:+.1%} |",
        f"| 负样本拒答率 | {naive['negative_reject_rate']:.1%} | {opt['negative_reject_rate']:.1%} | {delta['negative_reject_rate']:+.1%} |",
        f"| 表格数值准确率 | {naive['table_value_accuracy']:.1%} | {opt['table_value_accuracy']:.1%} | {delta.get('table_value_accuracy', 0):+.1%} |",
        "",
        "## 指标说明",
        "",
        "- **数值准确率**: expected_values 在答案中的命中率（归一化后字符串匹配）",
        "- **条件覆盖率**: expected_conditions 在答案中的覆盖率（核心关键词 60% 匹配）",
        "- **综合幻觉率**: (缺失值 + 缺失条件 + C3 校验幻觉数) / 总检查项",
        "- **负样本拒答率**: 负样本正确触发兜底的比例",
        "- **表格数值准确率**: 仅表格类问题的数值准确率",
        "",
        "## 优化措施",
        "",
        "1. **C2 约束型 Prompt**: 条件句逐条核对 + 表格数值原样引用 + 兜底转接",
        "2. **C3 原文事实校验**: 后置校验数值/条件是否在原文中有支撑",
        "3. **D1 无依据兜底**: 召回为空 / 超范围问题 / 校验失败时统一兜底",
        "",
    ]

    # 逐题明细：挑幻觉最严重的 10 条
    bad_cases = sorted(
        pos_results,
        key=lambda r: (
            (r["total_values"] - r["optimized_value_hits"])
            + (r["total_conditions"] - r["optimized_condition_hits"])
            + r.get("optimized_hallucination_count", 0)
        ),
        reverse=True,
    )[:10]

    if bad_cases:
        lines.extend([
            "## 优化后仍存在的问题（Top 10）",
            "",
        ])
        for r in bad_cases:
            missing_vals = [v for v in r.get("expected_values", []) if not check_value_in_answer(v, r["optimized_answer"])]
            missing_conds = [c for c in r.get("expected_conditions", []) if not check_condition_in_answer(c, r["optimized_answer"])]
            if not missing_vals and not missing_conds and r.get("optimized_hallucination_count", 0) == 0:
                continue
            lines.append(f"### Q{r['id']}: {r['question']}")
            if missing_vals:
                lines.append(f"- 缺失数值: {missing_vals}")
            if missing_conds:
                lines.append(f"- 缺失条件: {missing_conds}")
            if r.get("optimized_hallucination_count", 0) > 0:
                lines.append(f"- C3 校验幻觉: {r['optimized_hallucination_count']} 条")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
