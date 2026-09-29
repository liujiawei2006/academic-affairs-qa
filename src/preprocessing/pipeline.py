"""完整的预处理流水线：加载 → 清洗 → 结构标记 → 元数据提取 → 切分。"""
from pathlib import Path

from src.preprocessing.cleaners import TextCleaner
from src.preprocessing.chunkers import DocumentChunker, StructuredChunker
from src.preprocessing.loaders import DocumentLoader
from src.preprocessing.markers import StructureMarker
from src.preprocessing.metadata import MetadataExtractor
from src.preprocessing.models import DocumentChunk


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
