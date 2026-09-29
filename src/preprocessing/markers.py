"""结构化信息标记器：识别标题、表格、代码块。"""
import re


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
