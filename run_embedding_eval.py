"""Embedding 模型对比评估脚本。

阶段 4：双模型建库 + 同一评估入口跑两套，记录 Top5 召回 + API 延迟/成本。
候选模型：BAAI/bge-large-zh-v1.5（当前基线） vs BAAI/bge-m3。

使用方式：
    python run_embedding_eval.py
    python run_embedding_eval.py --skip-build          # 跳过 bge-m3 建库（collection 已存在时）
    python run_embedding_eval.py --rebuild-bge-large   # 同时重建基线库

产出物：
    docs/eval_results/embedding_bge_large/（基线模型评估结果）
    docs/eval_results/embedding_bge_m3/（候选模型评估结果）
    docs/eval_results/embedding_comparison/report.md（对比报告）
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import evaluate_recall, load_golden_dataset, _save_to_eval_results
from src.vectorstore import build_vector_store, load_vector_store
from src.preprocessing import DocumentChunk

# 模型配置
MODELS = {
    "bge_large": {
        "label": "embedding_bge_large",
        "model": "BAAI/bge-large-zh-v1.5",
        "collection": "academic_affairs",
        "max_token": 512,
        "pricing": "免费",
    },
    "bge_m3": {
        "label": "embedding_bge_m3",
        "model": "BAAI/bge-m3",
        "collection": "academic_affairs_bge_m3",
        "max_token": 8192,
        "pricing": "免费",
    },
}


def load_chunks(chunks_path: str) -> list[DocumentChunk]:
    """从 chunks.json 加载文档片段，转为 DocumentChunk 列表。"""
    with open(chunks_path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        DocumentChunk(content=r["text"], metadata=r.get("metadata", {}))
        for r in records
    ]


def build_collection(model_key: str, chunks: list[DocumentChunk]) -> float:
    """构建指定模型的 collection，返回建库耗时（秒）。"""
    config = MODELS[model_key]
    print(f"\n{'=' * 50}")
    print(f"  构建 collection: {config['collection']}")
    print(f"  模型: {config['model']}")
    print(f"{'=' * 50}")

    start = time.time()
    build_vector_store(
        chunks,
        drop_old=True,
        collection_name=config["collection"],
        embedding_model=config["model"],
    )
    build_time = time.time() - start
    print(f"建库耗时: {build_time:.1f}s")
    return build_time


def eval_model(model_key: str, dataset_path: str) -> dict:
    """对指定模型跑评估，返回召回结果 + 评估耗时。"""
    config = MODELS[model_key]
    print(f"\n{'=' * 50}")
    print(f"  评估模型: {config['model']}")
    print(f"  collection: {config['collection']}")
    print(f"{'=' * 50}")

    vectorstore = load_vector_store(
        collection_name=config["collection"],
        embedding_model=config["model"],
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    dataset = load_golden_dataset(dataset_path)
    print(f"加载测试集: {len(dataset)} 条")

    start = time.time()
    result = evaluate_recall(retriever, dataset, top_k_list=[1, 3, 5], max_fetch_k=10)
    eval_time = time.time() - start

    return {
        "recall": result["summary"],
        "per_question": result["per_question"],
        "total_questions": result["total_questions"],
        "table_questions": result["table_questions"],
        "eval_time_s": round(eval_time, 2),
    }


def generate_comparison_report(results: dict, build_times: dict, output_path: str):
    """生成 Markdown 对比报告。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Embedding 模型对比评估报告",
        "",
        "## 模型配置",
        "",
        "| 配置项 | bge-large-zh-v1.5 | bge-m3 |",
        "|--------|:---:|:---:|",
        f"| 模型 | {MODELS['bge_large']['model']} | {MODELS['bge_m3']['model']} |",
        "| 向量维度 | 1024 | 1024 |",
        f"| max token | {MODELS['bge_large']['max_token']} | {MODELS['bge_m3']['max_token']} |",
        f"| collection 名 | {MODELS['bge_large']['collection']} | {MODELS['bge_m3']['collection']} |",
        f"| SiliconFlow 定价 | {MODELS['bge_large']['pricing']} | {MODELS['bge_m3']['pricing']} |",
        "",
        "## 召回率对比",
        "",
        "| 指标 | bge-large-zh-v1.5 | bge-m3 | 差异 |",
        "|------|:---:|:---:|:---:|",
    ]

    r_large = results["bge_large"]["recall"]
    r_m3 = results["bge_m3"]["recall"]

    metric_labels = {
        "top1_recall": "Top1 整体",
        "top3_recall": "Top3 整体",
        "top5_recall": "Top5 整体",
        "top1_table_recall": "Top1 表格",
        "top3_table_recall": "Top3 表格",
        "top5_table_recall": "Top5 表格",
    }

    for metric_key, label in metric_labels.items():
        v_large = r_large.get(metric_key, 0)
        v_m3 = r_m3.get(metric_key, 0)
        diff = v_m3 - v_large
        diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
        lines.append(f"| {label} | {v_large:.1%} | {v_m3:.1%} | {diff_str} |")

    # 延迟对比
    bt_large = build_times.get("bge_large")
    bt_m3 = build_times.get("bge_m3")
    et_large = results["bge_large"]["eval_time_s"]
    et_m3 = results["bge_m3"]["eval_time_s"]

    bt_large_str = f"{bt_large:.1f}s" if bt_large else "(已有库)"
    bt_m3_str = f"{bt_m3:.1f}s" if bt_m3 else "(跳过)"

    lines.extend([
        "",
        "## 延迟对比",
        "",
        "| 指标 | bge-large-zh-v1.5 | bge-m3 |",
        "|------|:---:|:---:|",
        f"| 建库耗时 | {bt_large_str} | {bt_m3_str} |",
        f"| 评估耗时(含查询编码) | {et_large:.1f}s | {et_m3:.1f}s |",
        "",
        "## 选型结论",
        "",
        "_根据上方数据填写：_",
        "- 召回质量：哪个模型整体/表格子集更优？",
        "- 延迟：bge-m3 是否因参数量更大而明显更慢？",
        "- 成本：两者在 SiliconFlow 标准版均免费，成本无差异",
        "- 最终推荐：综合考虑召回提升 vs 延迟代价",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Embedding 模型对比评估")
    parser.add_argument("--dataset", type=str, default="tests/golden_dataset.jsonl")
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.json")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过 bge-m3 建库（假设 collection 已存在）")
    parser.add_argument("--rebuild-bge-large", action="store_true",
                        help="同时重建 bge-large-zh 基线库")
    args = parser.parse_args()

    print("=" * 60)
    print("   Embedding 模型对比评估")
    print("   bge-large-zh-v1.5 vs bge-m3")
    print("=" * 60)

    build_times = {}

    # 1. 建库
    need_build = not args.skip_build or args.rebuild_bge_large
    if need_build:
        chunks_path = args.chunks
        if not Path(chunks_path).exists():
            print(f"chunks 文件不存在: {chunks_path}")
            print("请先运行预处理生成 chunks.json")
            return
        chunks = load_chunks(chunks_path)
        print(f"加载 {len(chunks)} 个 chunk")

    if args.rebuild_bge_large:
        build_times["bge_large"] = build_collection("bge_large", chunks)

    if not args.skip_build:
        build_times["bge_m3"] = build_collection("bge_m3", chunks)

    # 2. 评估两个模型
    results = {}
    for model_key in MODELS:
        results[model_key] = eval_model(model_key, args.dataset)

        # 单独保存每个模型的评估结果
        label = MODELS[model_key]["label"]
        report = {
            "label": label,
            "model": MODELS[model_key]["model"],
            **results[model_key],
        }
        _save_to_eval_results(report, label, "eval_results")

    # 3. 生成对比报告
    comparison_path = "docs/eval_results/embedding_comparison/report.md"
    generate_comparison_report(results, build_times, comparison_path)

    # 4. 打印摘要
    print("\n" + "=" * 60)
    print("   对比摘要")
    print("=" * 60)
    r_large = results["bge_large"]["recall"]
    r_m3 = results["bge_m3"]["recall"]
    for key in ["top1_recall", "top5_recall", "top1_table_recall", "top5_table_recall"]:
        v_l = r_large.get(key, 0)
        v_m = r_m3.get(key, 0)
        diff = v_m - v_l
        print(f"  {key}: {v_l:.1%} -> {v_m:.1%} ({diff:+.1%})")

    bt_m3 = build_times.get("bge_m3")
    bt_m3_str = f"{bt_m3:.1f}s" if bt_m3 else "(跳过)"
    print(f"\n建库耗时: bge-m3 = {bt_m3_str}")
    print(f"评估耗时: bge-large = {results['bge_large']['eval_time_s']:.1f}s, "
          f"bge-m3 = {results['bge_m3']['eval_time_s']:.1f}s")
    print(f"\n结果已落盘: docs/eval_results/embedding_comparison/")


if __name__ == "__main__":
    main()
