"""脏数据清洗器：去除页眉页脚、冗余空白、乱码字符。"""
import re


class TextCleaner:
    """脏数据清洗器：去除页眉页脚、冗余空白、乱码字符。"""

    # 常见页眉页脚模式
    HEADER_FOOTER_PATTERNS = [
        r"^第\s*\d+\s*页.*$",  # 第 X 页
        r"^Page\s*\d+.*$",  # Page X
        r"^-\s*\d+\s*-$",  # - X -
        r"^\d+\s*/\s*\d+$",  # X / Y
        r"^共\s*\d+\s*页.*$",  # 共 X 页
    ]

    # 乱码字符模式（控制字符、不可见字符）
    GARBAGE_PATTERNS = [
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",  # 控制字符
        r"[\ufeff\ufffe\uffff]",  # BOM 和非法字符
        r"[\u200b-\u200f\u2028\u2029]",  # 零宽字符
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """完整清洗流程。"""
        # 1. 去除乱码字符
        text = cls._remove_garbage(text)

        # 2. 去除页眉页脚
        text = cls._remove_header_footer(text)

        # 3. 规范化空白
        text = cls._normalize_whitespace(text)

        return text.strip()

    @classmethod
    def _remove_garbage(cls, text: str) -> str:
        """去除乱码和控制字符。"""
        for pattern in cls.GARBAGE_PATTERNS:
            text = re.sub(pattern, "", text)
        return text

    @classmethod
    def _remove_header_footer(cls, text: str) -> str:
        """去除页眉页脚行。"""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # 检查是否匹配页眉页脚模式
            is_header_footer = any(
                re.match(pattern, stripped, re.IGNORECASE)
                for pattern in cls.HEADER_FOOTER_PATTERNS
            )
            if not is_header_footer:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        """规范化空白字符。"""
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 去除行首行尾空白（保留缩进结构）
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 合并多个空行（最多保留 2 个）
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        # 合并多个空格（表格内容除外）
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if not line.strip().startswith("|"):
                line = re.sub(r"[ \t]{2,}", " ", line)
            cleaned.append(line)
        text = "\n".join(cleaned)

        return text
