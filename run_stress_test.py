"""压测脚本：10 并发请求 /qa 接口，采集耗时 / Token / 成功率，输出压测报告。

用法：
    # 确保 API 服务已启动（python run_api.py）
    python run_stress_test.py

参数（可通过命令行覆盖）：
    --concurrency  并发数（默认 10）
    --total        总请求数（默认 20）
    --dataset      测试数据集路径（默认 tests/golden_dataset.jsonl）
    --api-url      API 地址（默认 http://localhost:8000/qa）
    --output       报告输出路径（默认 docs/stress_test_report.md）

排除 negative 样本（id 87-94），仅测试知识库覆盖范围内的正常问题。
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx


def load_questions(dataset_path: str, count: int) -> list[dict]:
    """从 golden_dataset.jsonl 中随机抽取指定数量的正常问题（排除 negative 样本）。"""
    questions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 排除 negative 样本（超范围问题，会触发兜底）
            if item.get("negative_sample"):
                continue
            questions.append(item)

    # 随机抽取
    sample_size = min(count, len(questions))
    sampled = random.sample(questions, sample_size)
    print(f"从 {len(questions)} 条正常问题中抽取 {sample_size} 条")
    return sampled


async def send_request(
    client: httpx.AsyncClient,
    api_url: str,
    question: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """发送单个请求，采集指标。"""
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await client.post(
                api_url,
                json={"question": question["question"], "enable_verify": False},
                timeout=120,
            )
            elapsed = time.perf_counter() - start

            if resp.status_code == 200:
                data = resp.json()
                timing = data.get("timing", {})
                token_usage = data.get("token_usage", {})
                return {
                    "id": question["id"],
                    "question": question["question"],
                    "status_code": resp.status_code,
                    "total_s": round(elapsed, 3),
                    "retrieval_s": timing.get("retrieval_s", 0),
                    "context_build_s": timing.get("context_build_s", 0),
                    "llm_s": timing.get("llm_s", 0),
                    "verify_s": timing.get("verify_s", 0),
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                    "citation_count": len(data.get("citations", [])),
                    "success": True,
                    "error": None,
                }
            else:
                return {
                    "id": question["id"],
                    "question": question["question"],
                    "status_code": resp.status_code,
                    "total_s": round(elapsed, 3),
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
                }
        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "id": question["id"],
                "question": question["question"],
                "status_code": 0,
                "total_s": round(elapsed, 3),
                "success": False,
                "error": str(e),
            }


async def run_stress_test(
    api_url: str,
    dataset_path: str,
    concurrency: int,
    total: int,
) -> list[dict]:
    """执行压测，返回所有请求的指标结果。"""
    questions = load_questions(dataset_path, total)
    semaphore = asyncio.Semaphore(concurrency)

    print(f"\n开始压测：并发={concurrency}，总请求={len(questions)}")
    print(f"API 地址：{api_url}")
    print("-" * 60)

    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, api_url, q, semaphore) for q in questions]
        results = await asyncio.gather(*tasks)

    # 按 id 排序
    results.sort(key=lambda r: r["id"])
    return results


def compute_percentile(values: list[float], p: float) -> float:
    """计算百分位数。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def generate_report(results: list[dict], concurrency: int, output_path: str):
    """生成压测报告（Markdown 格式）。"""
    success_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    total = len(results)
    success_count = len(success_results)
    fail_count = len(failed_results)
    success_rate = success_count / total * 100 if total else 0

    # 耗时统计
    latencies = [r["total_s"] for r in success_results]
    retrieval_times = [r["retrieval_s"] for r in success_results]
    llm_times = [r["llm_s"] for r in success_results]

    # Token 统计
    total_tokens_list = [r["total_tokens"] for r in success_results]
    prompt_tokens_list = [r["prompt_tokens"] for r in success_results]
    completion_tokens_list = [r["completion_tokens"] for r in success_results]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# 压测报告",
        f"",
        f"生成时间：{now}",
        f"",
        f"## 测试配置",
        f"",
        f"| 参数 | 值 |",
        f"|------|------|",
        f"| 并发数 | {concurrency} |",
        f"| 总请求数 | {total} |",
        f"| 成功数 | {success_count} |",
        f"| 失败数 | {fail_count} |",
        f"| 成功率 | {success_rate:.1f}% |",
        f"",
        f"## 响应耗时（秒）",
        f"",
        f"| 指标 | 总耗时 | 检索 | LLM 生成 |",
        f"|------|--------|------|----------|",
    ]

    if latencies:
        lines.extend([
            f"| 平均值 | {statistics.mean(latencies):.3f} | {statistics.mean(retrieval_times):.3f} | {statistics.mean(llm_times):.3f} |",
            f"| 中位数 (P50) | {compute_percentile(latencies, 50):.3f} | {compute_percentile(retrieval_times, 50):.3f} | {compute_percentile(llm_times, 50):.3f} |",
            f"| P95 | {compute_percentile(latencies, 95):.3f} | {compute_percentile(retrieval_times, 95):.3f} | {compute_percentile(llm_times, 95):.3f} |",
            f"| P99 | {compute_percentile(latencies, 99):.3f} | {compute_percentile(retrieval_times, 99):.3f} | {compute_percentile(llm_times, 99):.3f} |",
            f"| 最小值 | {min(latencies):.3f} | {min(retrieval_times):.3f} | {min(llm_times):.3f} |",
            f"| 最大值 | {max(latencies):.3f} | {max(retrieval_times):.3f} | {max(llm_times):.3f} |",
        ])

    lines.extend([
        f"",
        f"## Token 消耗",
        f"",
        f"| 指标 | prompt_tokens | completion_tokens | total_tokens |",
        f"|------|---------------|-------------------|--------------|",
    ])

    if total_tokens_list:
        lines.extend([
            f"| 平均值 | {statistics.mean(prompt_tokens_list):.0f} | {statistics.mean(completion_tokens_list):.0f} | {statistics.mean(total_tokens_list):.0f} |",
            f"| 总计 | {sum(prompt_tokens_list)} | {sum(completion_tokens_list)} | {sum(total_tokens_list)} |",
        ])

    lines.extend([
        f"",
        f"## 逐请求明细",
        f"",
        f"| ID | 问题 | 状态 | 耗时(s) | 检索(s) | LLM(s) | Tokens | 引用数 |",
        f"|----|------|------|---------|---------|--------|--------|--------|",
    ])

    for r in results:
        q_short = r["question"][:20] + "..." if len(r["question"]) > 20 else r["question"]
        if r["success"]:
            lines.append(
                f"| {r['id']} | {q_short} | {r['status_code']} | "
                f"{r['total_s']:.3f} | {r['retrieval_s']:.3f} | {r['llm_s']:.3f} | "
                f"{r['total_tokens']} | {r['citation_count']} |"
            )
        else:
            lines.append(
                f"| {r['id']} | {q_short} | FAIL | "
                f"{r['total_s']:.3f} | - | - | - | - |"
            )

    if failed_results:
        lines.extend([
            f"",
            f"## 失败请求",
            f"",
        ])
        for r in failed_results:
            lines.append(f"- ID {r['id']}: {r['error']}")

    lines.extend([
        f"",
        f"## 瓶颈分析",
        f"",
    ])

    if latencies:
        avg_llm_ratio = statistics.mean(llm_times) / statistics.mean(latencies) * 100 if statistics.mean(latencies) > 0 else 0
        avg_retrieval_ratio = statistics.mean(retrieval_times) / statistics.mean(latencies) * 100 if statistics.mean(latencies) > 0 else 0
        lines.extend([
            f"- **LLM 生成**占总耗时 {avg_llm_ratio:.0f}%（主要瓶颈）",
            f"- **检索**占总耗时 {avg_retrieval_ratio:.0f}%",
            f"- 其余为 context 构建、序列化、网络传输等",
            f"",
            f"### 优化建议",
            f"",
            f"1. LLM 调用是主要瓶颈，可考虑：流式输出（减少用户等待感）、使用更快的模型、缓存高频问题",
            f"2. 检索链路已包含 Hybrid + Rerank，延迟在可接受范围",
            f"3. 并发场景下关注 API 限流和 Token 配额",
        ])

    report_text = "\n".join(lines)

    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n压测报告已保存: {output}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="教务问答 API 压测脚本")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数（默认 10）")
    parser.add_argument("--total", type=int, default=20, help="总请求数（默认 20）")
    parser.add_argument("--dataset", type=str, default="tests/golden_dataset.jsonl", help="测试数据集路径")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/qa", help="API 地址")
    parser.add_argument("--output", type=str, default="docs/stress_test_report.md", help="报告输出路径")
    args = parser.parse_args()

    # 运行压测
    results = asyncio.run(run_stress_test(args.api_url, args.dataset, args.concurrency, args.total))

    # 生成报告
    report = generate_report(results, args.concurrency, args.output)

    # 终端摘要
    success = [r for r in results if r["success"]]
    if success:
        latencies = [r["total_s"] for r in success]
        print(f"\n{'=' * 60}")
        print(f"压测完成：成功 {len(success)}/{len(results)}")
        print(f"耗时 P50={compute_percentile(latencies, 50):.3f}s  "
              f"P95={compute_percentile(latencies, 95):.3f}s  "
              f"P99={compute_percentile(latencies, 99):.3f}s")
        total_tokens = sum(r["total_tokens"] for r in success)
        print(f"Token 总计: {total_tokens}")


if __name__ == "__main__":
    main()
