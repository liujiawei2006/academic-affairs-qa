"""增量导入脚本：将新文件导入到指定学年。

用法：
    python run_import.py --dir data/raw/new_files --academic-year 2025-2026

参数：
    --dir            新文件所在目录（必填）
    --academic-year  学年标签，如 "2025-2026"（必填）
    --list-years     列出当前所有学年

示例：
    # 查看当前学年
    python run_import.py --list-years

    # 导入新文件
    python run_import.py --dir data/raw/2025-2026新政策 --academic-year 2025-2026
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="增量导入新文件到知识库")
    parser.add_argument("--dir", type=str, help="新文件所在目录")
    parser.add_argument("--academic-year", type=str, help="学年标签，如 2025-2026")
    parser.add_argument("--list-years", action="store_true", help="列出当前所有学年")
    args = parser.parse_args()

    from src.versioning import import_files, list_academic_years

    # 列出学年
    if args.list_years:
        print("=== 当前知识库学年列表 ===")
        years = list_academic_years()
        if not years:
            print("（空）")
        for y in years:
            print(f"  {y['academic_year']}: {y['count']} 条 chunk")
        return

    # 增量导入
    if not args.dir or not args.academic_year:
        parser.error("--dir 和 --academic-year 必须同时指定")

    print(f"=== 增量导入 ===")
    print(f"目录: {args.dir}")
    print(f"学年: {args.academic_year}")
    print()

    count = import_files(args.dir, args.academic_year)
    print(f"\n导入完成: {count} 条 chunk")


if __name__ == "__main__":
    main()
