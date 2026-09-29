"""元数据提取器：来源、页码、更新时间、业务标签。"""
import re
from datetime import datetime
from pathlib import Path


class MetadataExtractor:
    """元数据提取器：来源、页码、更新时间、业务标签。"""

    # 业务标签关键词映射
    BUSINESS_TAG_KEYWORDS = {
        "奖学金": ["奖学金", "助学金", "励志", "评优"],
        "选课": ["选课", "课程", "学分", "必修", "选修", "限选"],
        "学籍": ["学籍", "注册", "休学", "复学", "退学", "开除"],
        "转专业": ["转专业", "专业分流", "转学"],
        "培养方案": ["培养方案", "教学计划", "课程设置", "毕业要求", "培养计划"],
        "考试": ["考试", "补考", "重修", "缓考", "作弊", "违纪"],
        "创新学分": ["创新学分", "创新分", "学科竞赛"],
        "勤工助学": ["勤工助学", "助学岗位"],
        "综合测评": ["综合测评", "综测"],
        "英语课程": ["英语", "四六级", "免修", "重考"],
    }

    # 业务标签优先级：越靠前越具体，通用标签（选课）放最后。
    # 用于文件名命中时的优先选择，以及正文评分同分时的 tie-break。
    BUSINESS_TAG_PRIORITY = [
        "转专业", "创新学分", "勤工助学", "综合测评", "英语课程",
        "培养方案", "奖学金", "考试", "学籍", "选课",
    ]

    # 强特征词：在正文中出现即可定性该文档所属标签，
    # 优先级高于"关键词计数评分"，避免培养方案文档因"学分/课程"高频而被误标为选课。
    BUSINESS_TAG_STRONG_MARKERS = {
        "培养方案": ["培养目标", "毕业要求", "培养计划", "培养方案", "主干学科"],
        "转专业": ["转专业", "专业分流"],
        "创新学分": ["创新学分", "创新分"],
        "勤工助学": ["勤工助学"],
        "综合测评": ["综合测评", "综测"],
        "奖学金": ["奖学金", "助学金"],
    }

    @classmethod
    def extract(cls, file_path: str, content: str, loader_metadata: dict = None) -> dict:
        """提取完整元数据。"""
        file_path_obj = Path(file_path)
        filename = file_path_obj.stem

        # 基础元数据
        metadata = {
            "source_file": file_path_obj.name,
            "source_path": str(file_path_obj),
            "file_type": file_path_obj.suffix.lower(),
            "business_tag": cls._extract_business_tag(filename, content),
            "academic_year": cls._extract_academic_year(filename, content),
            "chapter": cls._extract_chapter(content),
            "update_time": cls._extract_update_time(file_path_obj, content),
            "created_at": datetime.now().isoformat(),
        }

        # 合并加载器提供的元数据（如页码信息）
        if loader_metadata:
            metadata.update(loader_metadata)

        return metadata

    @classmethod
    def _extract_business_tag(cls, filename: str, content: str) -> str:
        """提取业务标签。

        策略：
        1. 文件名优先——文件名是最强信号，按优先级顺序匹配，命中即返回；
        2. 正文评分兜底——文件名无命中时，统计各标签关键词在正文出现次数取最高分，
           同分时按优先级取更具体的标签，避免"选课"这类通用标签抢占"培养方案"。
        """
        # 1. 文件名优先匹配（按优先级顺序）
        for tag in cls.BUSINESS_TAG_PRIORITY:
            keywords = cls.BUSINESS_TAG_KEYWORDS.get(tag, [])
            if any(kw in filename for kw in keywords):
                return tag

        body = content[:2000]

        # 2. 正文强特征词定性（按优先级顺序，命中即返回）
        for tag in cls.BUSINESS_TAG_PRIORITY:
            markers = cls.BUSINESS_TAG_STRONG_MARKERS.get(tag, [])
            if any(mk in body for mk in markers):
                return tag

        # 3. 正文关键词评分兜底
        best_tag = "其他"
        best_score = 0
        for tag in cls.BUSINESS_TAG_PRIORITY:
            keywords = cls.BUSINESS_TAG_KEYWORDS.get(tag, [])
            score = sum(body.count(kw) for kw in keywords)
            if score > best_score:
                best_score = score
                best_tag = tag
        return best_tag if best_score > 0 else "其他"

    @classmethod
    def _extract_academic_year(cls, filename: str, content: str) -> str:
        """提取学年信息。"""
        patterns = [
            r"(\d{4})[-—](\d{4})",  # 2024-2025
            r"(\d{4})\s*年",  # 2024年
            r"(\d{4})级",  # 2024级
            r"(\d{4})",  # 2024
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if match.lastindex == 2:
                    return f"{match.group(1)}-{match.group(2)}"
                return match.group(1)

        for pattern in patterns:
            match = re.search(pattern, content[:500])
            if match:
                if match.lastindex == 2:
                    return f"{match.group(1)}-{match.group(2)}"
                return match.group(1)

        return ""

    @classmethod
    def _extract_chapter(cls, content: str) -> str:
        """提取第一个标题作为章节。"""
        match = re.search(r"^#+\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    @classmethod
    def _extract_update_time(cls, file_path: Path, content: str) -> str:
        """提取更新时间（优先从内容，其次从文件修改时间）。"""
        # 从内容中查找日期
        date_patterns = [
            r"(\d{4})年(\d{1,2})月(\d{1,2})日",
            r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content[:1000])
            if match:
                year, month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"

        # 使用文件修改时间
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
