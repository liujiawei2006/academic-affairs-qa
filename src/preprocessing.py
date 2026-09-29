"""文档预处理流水线：多格式加载 + 脏数据清洗 + 结构化保留 + 元数据标注。"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pdfplumber
import requests
from bs4 import BeautifulSoup
from docx import Document


@dataclass
class DocumentChunk:
    """标准化的文档片段，带完整元数据。"""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "")

    @property
    def business_tag(self) -> str:
        return self.metadata.get("business_tag", "")

    @property
    def academic_year(self) -> str:
        return self.metadata.get("academic_year", "")

    @property
    def chapter(self) -> str:
        return self.metadata.get("chapter", "")

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number")

    @property
    def content_type(self) -> str:
        return self.metadata.get("content_type", "text")


class DocumentLoader:
    """多格式文档加载器：PDF/Word/Markdown/网页/纯文本。"""

    # OCR 引擎懒加载单例（首次遇到无文字层的扫描件时才初始化）
    _ocr_engine = None
    OCR_RENDER_DPI = 200  # 扫描件页面转图片的渲染分辨率

    @classmethod
    def _get_ocr_engine(cls):
        """懒加载 RapidOCR 引擎，未安装时返回 None。"""
        if cls._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                cls._ocr_engine = RapidOCR()
            except ImportError:
                cls._ocr_engine = False  # 标记不可用，避免反复尝试导入
        return cls._ocr_engine or None

    @classmethod
    def _ocr_page(cls, page) -> str:
        """对无文字层的 PDF 页面做 OCR，返回识别出的文本。"""
        engine = cls._get_ocr_engine()
        if engine is None:
            print(f"⚠️  检测到扫描页但未安装 OCR，请执行: pip install rapidocr-onnxruntime")
            return ""
        img = page.to_image(resolution=cls.OCR_RENDER_DPI)
        pil_img = img.original  # PIL Image，RapidOCR 支持 ndarray 输入
        import numpy as np
        result, _ = engine(np.array(pil_img))
        if not result:
            return ""
        # result 格式: [[bbox, text, confidence], ...]，按识别顺序拼接
        return "\n".join(line[1] for line in result)

    @staticmethod
    def load_pdf(path: str) -> tuple[str, dict]:
        """加载 PDF，提取文本和表格（转 Markdown）；无文字层的扫描页自动走 OCR 兜底。"""
        content_parts = []
        page_map = []  # 记录每个内容块的页码
        ocr_pages = []  # 走了 OCR 的页码列表

        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_content = []

                # 提取表格并转 Markdown
                tables = page.extract_tables()
                for table in tables:
                    md_table = DocumentLoader._table_to_markdown(table)
                    if md_table:
                        page_content.append(md_table)

                # 提取普通文本
                text = page.extract_text()
                if text:
                    page_content.append(text)

                # 文字层为空但有图片 → 扫描件，走 OCR 兜底
                if not text and page.images:
                    ocr_text = DocumentLoader._ocr_page(page)
                    if ocr_text.strip():
                        page_content.append(ocr_text.strip())
                        ocr_pages.append(page_num)

                if page_content:
                    combined = "\n\n".join(page_content)
                    content_parts.append(combined)
                    page_map.append({"page": page_num, "length": len(combined)})

        full_content = "\n\n".join(content_parts)
        metadata = {"page_map": page_map, "total_pages": total_pages}
        if ocr_pages:
            metadata["ocr_pages"] = ocr_pages
        return full_content, metadata

    @staticmethod
    def load_docx(path: str) -> tuple[str, dict]:
        """加载 Word 文档，保留标题层级和表格。"""
        doc = Document(path)
        content_parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                # 根据标题级别转 Markdown
                if para.style.name.startswith("Heading"):
                    level = int(para.style.name.replace("Heading", "").strip() or "1")
                    content_parts.append(f"{'#' * level} {para.text}\n\n")
                else:
                    content_parts.append(f"{para.text}\n\n")

        # 提取表格
        for table in doc.tables:
            md_table = DocumentLoader._docx_table_to_markdown(table)
            if md_table:
                content_parts.append(md_table)
                content_parts.append("\n\n")

        return "".join(content_parts).strip(), {}

    @staticmethod
    def load_markdown(path: str) -> tuple[str, dict]:
        """加载 Markdown 文件。"""
        with open(path, encoding="utf-8") as f:
            return f.read(), {}

    @staticmethod
    def load_text(path: str) -> tuple[str, dict]:
        """加载纯文本文件。"""
        with open(path, encoding="utf-8") as f:
            return f.read(), {}

    @staticmethod
    def load_html(url_or_path: str) -> tuple[str, dict]:
        """加载网页或 HTML 文件，提取正文内容。"""
        if url_or_path.startswith(("http://", "https://")):
            resp = requests.get(url_or_path, timeout=30)
            resp.raise_for_status()
            html_content = resp.text
        else:
            with open(url_or_path, encoding="utf-8") as f:
                html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # 移除脚本和样式
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # 提取标题
        title = soup.title.string.strip() if soup.title else ""

        # 提取正文（优先 article/main 标签）
        main_content = soup.find("article") or soup.find("main") or soup.find("body")

        if main_content:
            # 保留标题层级
            for h in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                level = int(h.name[1])
                h.replace_with(f"\n{'#' * level} {h.get_text(strip=True)}\n\n")

            # 保留表格
            for table in main_content.find_all("table"):
                md_table = DocumentLoader._html_table_to_markdown(table)
                table.replace_with(f"\n{md_table}\n\n")

            # 保留代码块
            for code in main_content.find_all(["pre", "code"]):
                code_text = code.get_text()
                code.replace_with(f"\n```\n{code_text}\n```\n\n")

            text = main_content.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        metadata = {"title": title, "source_url": url_or_path}
        return text, metadata

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """将 pdfplumber 提取的表格转为 Markdown 格式。"""
        if not table or not table[0]:
            return ""

        # 清理单元格内容
        cleaned = []
        for row in table:
            cleaned_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
            cleaned.append(cleaned_row)

        # 生成 Markdown 表格
        lines = []
        lines.append("| " + " | ".join(cleaned[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(cleaned[0])) + " |")
        for row in cleaned[1:]:
            while len(row) < len(cleaned[0]):
                row.append("")
            lines.append("| " + " | ".join(row[:len(cleaned[0])]) + " |")

        return "\n".join(lines)

    @staticmethod
    def _docx_table_to_markdown(table) -> str:
        """将 python-docx 的表格转为 Markdown 格式。"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append(cells)

        if not rows:
            return ""

        lines = []
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _html_table_to_markdown(table) -> str:
        """将 HTML 表格转为 Markdown 格式。"""
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        lines = []
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @classmethod
    def load(cls, path: str) -> tuple[str, dict]:
        """根据文件扩展名自动选择加载器。"""
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return cls.load_pdf(path)
        elif ext == ".docx":
            return cls.load_docx(path)
        elif ext == ".md":
            return cls.load_markdown(path)
        elif ext in (".html", ".htm"):
            return cls.load_html(path)
        elif path.startswith(("http://", "https://")):
            return cls.load_html(path)
        else:
            return cls.load_text(path)


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


class StructureMarker:
    """结构化信息标记器：识别标题、表格、代码块。"""

    @staticmethod
    def mark_structures(text: str) -> str:
        """识别并标记文档结构。"""
        # 1. 标记代码块（如果还没有标记）
        text = StructureMarker._mark_code_blocks(text)

        # 2. 规范化表格格式
        text = StructureMarker._normalize_tables(text)

        # 3. 规范化标题格式
        text = StructureMarker._normalize_headings(text)

        return text

    @staticmethod
    def _mark_code_blocks(text: str) -> str:
        """识别代码块并添加 Markdown 标记。"""
        # 检测连续的等宽字体行（简单启发式：包含大量特殊字符）
        lines = text.split("\n")
        result = []
        in_code = False
        code_buffer = []

        for line in lines:
            # 代码行特征：包含代码符号或缩进
            is_code_line = bool(re.search(r"[{}();=<>]|^\s{4,}\S", line)) and len(line.strip()) > 0

            if is_code_line and not in_code:
                in_code = True
                code_buffer = [line]
            elif is_code_line and in_code:
                code_buffer.append(line)
            elif not is_code_line and in_code:
                # 结束代码块
                if len(code_buffer) >= 3:  # 至少 3 行才算代码块
                    result.append("```")
                    result.extend(code_buffer)
                    result.append("```")
                else:
                    result.extend(code_buffer)
                in_code = False
                code_buffer = []
                result.append(line)
            else:
                result.append(line)

        # 处理结尾
        if in_code and len(code_buffer) >= 3:
            result.append("```")
            result.extend(code_buffer)
            result.append("```")
        elif in_code:
            result.extend(code_buffer)

        return "\n".join(result)

    @staticmethod
    def _normalize_tables(text: str) -> str:
        """规范化 Markdown 表格格式。"""
        lines = text.split("\n")
        result = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            is_table_line = stripped.startswith("|") and stripped.endswith("|")

            if is_table_line:
                if not in_table:
                    in_table = True
                    result.append("")  # 表格前空行
                result.append(stripped)
            else:
                if in_table:
                    in_table = False
                    result.append("")  # 表格后空行
                result.append(line)

        return "\n".join(result)

    # 标题正文最大长度：超过则视为编号列表项/整句正文，而非标题
    MAX_HEADING_TITLE_LEN = 40

    @staticmethod
    def _is_heading_like(title: str) -> bool:
        """判断"编号 + 文本"中的文本是否真的是标题，过滤编号列表项与整句正文。"""
        title = title.strip()
        # 标题一般较短；过长多为列表项或正文句子
        if not title or len(title) > StructureMarker.MAX_HEADING_TITLE_LEN:
            return False
        # 含句末标点说明是正文句子，而非标题
        if re.search(r"[。；！？]", title):
            return False
        return True

    @staticmethod
    def _normalize_headings(text: str) -> str:
        """规范化标题格式（过滤编号列表项误判）。"""
        lines = text.split("\n")
        result = []

        for line in lines:
            stripped = line.strip()

            # 检测中文标题模式（如 "一、" "第一章" "1.1"）
            heading_match = re.match(
                r"^(第[一二三四五六七八九十]+[章节]|"
                r"[一二三四五六七八九十]+、|"
                r"\d+(?:\.\d+)*[、.]\s*)"
                r"(.+)$",
                stripped
            )

            if heading_match and not stripped.startswith("#"):
                prefix = heading_match.group(1)
                title = heading_match.group(2)

                # 护栏：过长或含句末标点的"编号文本"视为正文，不当标题
                if not StructureMarker._is_heading_like(title):
                    result.append(line)
                    continue

                # 根据层级添加 Markdown 标记
                if prefix.startswith("第"):
                    result.append(f"## {prefix}{title}")
                elif re.match(r"^[一二三四五六七八九十]+、", prefix):
                    result.append(f"### {prefix}{title}")
                else:
                    result.append(f"#### {prefix}{title}")
            else:
                result.append(line)

        return "\n".join(result)

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
    # 优先级高于“关键词计数评分”，避免培养方案文档因“学分/课程”高频而被误标为选课。
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


class DocumentChunker:
    """文档切分器：保留表格和代码块完整性。"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, content: str, metadata: dict) -> list[DocumentChunk]:
        """切分文档，表格和代码块不拆分。"""
        # 按结构分割
        parts = self._split_by_structures(content)
        chunks = []

        for part in parts:
            if part["type"] == "table":
                # 表格整块保留
                if len(part["content"]) <= self.chunk_size * 2:
                    chunk_meta = {**metadata, "content_type": "table"}
                    chunks.append(DocumentChunk(part["content"], chunk_meta))
                else:
                    # 超长表格按行切分，保留表头
                    for tp in self._split_long_table(part["content"]):
                        chunk_meta = {**metadata, "content_type": "table"}
                        chunks.append(DocumentChunk(tp, chunk_meta))

            elif part["type"] == "code":
                # 代码块整块保留
                chunk_meta = {**metadata, "content_type": "code"}
                chunks.append(DocumentChunk(part["content"], chunk_meta))

            else:
                # 普通文本按段落切分
                for tc in self._split_text(part["content"]):
                    chunk_meta = {**metadata, "content_type": "text"}
                    chunks.append(DocumentChunk(tc, chunk_meta))

        return chunks

    def _split_by_structures(self, content: str) -> list[dict]:
        """按表格和代码块分割内容。"""
        parts = []
        lines = content.split("\n")
        current_text = []
        current_block = []
        block_type = None

        for line in lines:
            stripped = line.strip()

            # 检测表格行
            if stripped.startswith("|") and stripped.endswith("|"):
                if block_type != "table":
                    if current_text:
                        parts.append({"content": "\n".join(current_text), "type": "text"})
                        current_text = []
                    if current_block:
                        parts.append({"content": "\n".join(current_block), "type": block_type})
                        current_block = []
                    block_type = "table"
                current_block.append(line)
                continue

            # 检测代码块标记
            if stripped.startswith("```"):
                if block_type == "code":
                    # 结束代码块
                    current_block.append(line)
                    parts.append({"content": "\n".join(current_block), "type": "code"})
                    current_block = []
                    block_type = None
                else:
                    # 开始代码块
                    if current_text:
                        parts.append({"content": "\n".join(current_text), "type": "text"})
                        current_text = []
                    if current_block:
                        parts.append({"content": "\n".join(current_block), "type": block_type})
                        current_block = []
                    block_type = "code"
                    current_block.append(line)
                continue

            # 普通行
            if block_type == "code":
                current_block.append(line)
            elif block_type == "table":
                # 表格结束
                if current_block:
                    parts.append({"content": "\n".join(current_block), "type": "table"})
                    current_block = []
                block_type = None
                current_text.append(line)
            else:
                current_text.append(line)

        # 处理剩余内容
        if current_text:
            parts.append({"content": "\n".join(current_text), "type": "text"})
        if current_block and block_type:
            parts.append({"content": "\n".join(current_block), "type": block_type})

        return parts

    def _split_text(self, text: str) -> list[str]:
        """按段落切分文本，支持重叠窗口，超长段落硬切分。"""
        paragraphs = re.split(r"\n\n+", text.strip())
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 超长段落：先收尾当前块，再对该段落硬切分
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk).strip())
                    current_chunk = []
                    current_size = 0
                chunks.extend(self._hard_split(para))
                continue

            # 超出 chunk_size：收尾并按重叠窗口保留上一块尾部
            if current_size + len(para) > self.chunk_size and current_chunk:
                prev_text = "\n\n".join(current_chunk)
                chunks.append(prev_text.strip())
                if self.chunk_overlap > 0:
                    overlap = prev_text[-self.chunk_overlap:].strip()
                    current_chunk = [overlap] if overlap else []
                    current_size = len(overlap)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> list[str]:
        """对超长段落按 chunk_size 硬切分，保留重叠窗口。"""
        pieces = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        start = 0
        while start < len(text):
            piece = text[start:start + self.chunk_size]
            if piece.strip():
                pieces.append(piece)
            if start + self.chunk_size >= len(text):
                break
            start += step
        return pieces

    def _split_long_table(self, table_content: str) -> list[str]:
        """切分超长表格，保留表头。"""
        lines = table_content.split("\n")
        if len(lines) < 3:
            return [table_content]

        header = lines[:2]
        data_lines = lines[2:]

        chunks = []
        current_chunk = header.copy()
        current_size = sum(len(line) for line in header)

        for line in data_lines:
            line_size = len(line)
            if current_size + line_size > self.chunk_size and len(current_chunk) > 2:
                chunks.append("\n".join(current_chunk))
                current_chunk = header + [line]
                current_size = sum(len(l) for l in header) + line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if len(current_chunk) > 2:
            chunks.append("\n".join(current_chunk))

        return chunks


class StructuredChunker:
    """结构化（标题层级）切分器。

    切分逻辑：
    1. 按 Markdown 标题层级（# ~ ######）把文档拆成若干 section；
    2. 每个 section 记录完整的标题路径（面包屑），作为上下文前缀拼回片段，提升召回命中率；
    3. section 内部交给 DocumentChunker 做结构感知切分（表格/代码块不拆）；
    4. 元数据补充 heading_path / heading_level，便于溯源与过滤。
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, keep_heading_path: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_heading_path = keep_heading_path
        self._base_chunker = DocumentChunker(chunk_size, chunk_overlap)

    def chunk(self, content: str, metadata: dict) -> list[DocumentChunk]:
        """按标题层级切分文档。"""
        sections = self._parse_sections(content)
        chunks = []

        for sec in sections:
            heading_path = sec["heading_path"]
            body = sec["body"]

            # section 内部交给基础切分器（表格/代码块不拆）
            base_chunks = self._base_chunker.chunk(body, metadata)
            if not base_chunks:
                continue

            prefix = self._build_prefix(heading_path)
            breadcrumb = " > ".join(heading_path)

            for bc in base_chunks:
                text = prefix + bc.content if self.keep_heading_path else bc.content
                chunk_meta = {
                    **bc.metadata,
                    "chunk_strategy": "structured",
                    "heading_path": breadcrumb,
                    "heading_level": len(heading_path),
                }
                chunks.append(DocumentChunk(text, chunk_meta))

        return chunks

    def _parse_sections(self, content: str) -> list[dict]:
        """把文档按标题层级拆成 section，每个 section 带标题路径。"""
        lines = content.split("\n")
        sections = []
        heading_stack = []  # [(level, title), ...]
        current_headings = []
        current_body = []

        def _flush():
            body_text = "\n".join(current_body).strip()
            if body_text:
                sections.append({
                    "heading_path": [t for _, t in current_headings],
                    "body": body_text,
                })

        for line in lines:
            m = self.HEADING_RE.match(line.strip())
            if m:
                # 遇到新标题：先收尾上一个 section
                _flush()
                current_body = []

                level = len(m.group(1))
                title = m.group(2).strip()
                # 弹出同级或更低级标题，维护标题栈
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                current_headings = list(heading_stack)
            else:
                current_body.append(line)

        # 收尾最后一个 section
        _flush()
        return sections

    def _build_prefix(self, heading_path: list) -> str:
        """构造标题路径前缀，为片段补充上下文。"""
        if not heading_path or not self.keep_heading_path:
            return ""
        return " > ".join(heading_path) + "\n\n"


class PreprocessingPipeline:
    """完整的预处理流水线。"""

    SUPPORTED_STRATEGIES = {"structured", "recursive"}

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "structured",
    ):
        if chunk_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"不支持的切分策略: {chunk_strategy}，可选: {self.SUPPORTED_STRATEGIES}"
            )
        self.chunk_strategy = chunk_strategy
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.marker = StructureMarker()
        self.metadata_extractor = MetadataExtractor()
        if chunk_strategy == "structured":
            self.chunker = StructuredChunker(chunk_size, chunk_overlap)
        else:
            self.chunker = DocumentChunker(chunk_size, chunk_overlap)

    def process_file(self, file_path: str) -> list[DocumentChunk]:
        """处理单个文件。"""
        # 1. 加载（返回内容和加载器元数据）
        content, loader_metadata = self.loader.load(file_path)

        # 2. 清洗脏数据
        content = self.cleaner.clean(content)

        # 3. 标记结构化信息
        content = self.marker.mark_structures(content)

        # 4. 提取元数据
        metadata = self.metadata_extractor.extract(file_path, content, loader_metadata)

        # 5. 切分
        chunks = self.chunker.chunk(content, metadata)

        # 6. 统一补充切分策略标记与片段序号（结构化切分已在内部标记，此处兜底）
        for idx, c in enumerate(chunks):
            c.metadata.setdefault("chunk_strategy", self.chunk_strategy)
            c.metadata["chunk_index"] = idx

        print(f"✅ {Path(file_path).name}: {len(chunks)} 个片段（策略: {self.chunk_strategy}）")
        return chunks

    def process_directory(self, dir_path: str) -> list[DocumentChunk]:
        """处理目录下所有支持的文件。"""
        all_chunks = []
        supported_exts = {".pdf", ".docx", ".md", ".txt", ".html", ".htm"}

        for file_path in Path(dir_path).rglob("*"):
            if file_path.suffix.lower() in supported_exts:
                try:
                    chunks = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"⚠️  处理 {file_path} 失败: {e}")

        print(f"\n📊 总计: {len(all_chunks)} 个文档片段")
        return all_chunks
