"""文档片段数据模型：标准化输出结构，带完整元数据。"""
from dataclasses import dataclass, field
from typing import Optional


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
