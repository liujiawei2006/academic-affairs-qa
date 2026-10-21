"""向量重建/回滚脚本：按学年重建或删除知识库数据。

用法：
    # 重建某学年（先删后导）
    python run_rebuild.py --rebuild 2025-2026 --dir data/raw/2025-2026新政策

    # 回滚（删除）某学年
    python run_rebuild.py --rollback 2025-2026

    # 列出所有学年
    python run_rebuild.py --list-years

    # 统计某学年
    python run_rebuild.py --count 2025-2026

参数：
    --rebuild YEAR   重建指定学年（需配合 --dir）
    --rollback YEAR  回滚（删除）指定学年
    --count YEAR     统计指定学年的 chunk 数量
    --dir PATH       重建时的源文件目录
    --list-years     列出所有学年
    --force          跳过确认提示
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="知识库版本管理：重建/回滚")
    parser.add_argument("--rebuild", type=str, metavar="YEAR", help="重建指定学年")
    parser.add_argument("--rollback", type=str, metavar="YEAR", help="回滚（删除）指定学年")
    parser.add_argument("--count", type=str, metavar="YEAR", help="统计指定学年的 chunk 数量")
    parser.add_argument("--dir", type=str, help="重建时的源文件目录")
    parser.add_argument("--list-years", action="store_true", help="列出所有学年")
    parser.add_argument("--force", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    from src.versioning import (
        count_by_academic_year,
        delete_by_academic_year,
        list_academic_years,
        rebuild_academic_year,
    )

    # 列出学年
    if args.list_years:
        print("=== 当前知识库学年列表 ===")
        years = list_academic_years()
        if not years:
            print("（空）")
        for y in years:
            print(f"  {y['academic_year']}: {y['count']} 条 chunk")
        return

    # 统计
    if args.count:
        count = count_by_academic_year(args.count)
        print(f"学年 {args.count}: {count} 条 chunk")
        return

    # 重建
    if args.rebuild:
        if not args.dir:
            parser.error("--rebuild 必须配合 --dir 指定源文件目录")
        rebuild_academic_year(args.rebuild, args.dir, force=args.force)
        return

    # 回滚
    if args.rollback:
        deleted = delete_by_academic_year(args.rollback, force=args.force)
        print(f"回滚完成: 删除 {deleted} 条 chunk")
        return

    # 无操作
    parser.print_help()


if __name__ == "__main__":
    main()
