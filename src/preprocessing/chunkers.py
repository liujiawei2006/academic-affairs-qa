"""文档切分器：基础结构感知切分 + 标题层级结构化切分。"""
import re

from src.preprocessing.models import DocumentChunk


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
