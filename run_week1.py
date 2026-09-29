"""第 1 周任务执行脚本：预处理 + 向量库 + 评估。"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import PreprocessingPipeline
from src.vectorstore import build_vector_store, load_vector_store
from src.evaluation import run_full_evaluation


def main():
    print("=" * 60)
    print("   第 1 周任务：立标尺——基线系统 + 评估体系")
    print("=" * 60)

    # 1. 文档预处理
    print("\n📂 步骤 1: 文档预处理")
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"⚠️  {raw_dir} 目录为空，请先放入教务文档（PDF/Word/Markdown）")
        print("   处理完成后重新运行本脚本")
        return

    pipeline = PreprocessingPipeline(chunk_size=500, chunk_overlap=50, chunk_strategy="structured")
    chunks = pipeline.process_directory(raw_dir)

    if not chunks:
        print("❌ 未生成任何文档片段，请检查文档格式")
        return

    print(f"\n✅ 共生成 {len(chunks)} 个文档片段")

    # 2. 构建向量库
    print("\n📊 步骤 2: 构建向量库")
    vector_store = build_vector_store(chunks, drop_old=True)
    print(f"✅ 向量库构建完成，共 {len(chunks)} 条记录")

    # 3. 运行评估
    print("\n 步骤 3: 运行评估")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    report = run_full_evaluation(retriever)

    print("\n" + "=" * 60)
    print("   评估完成！报告已保存至 docs/baseline_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
