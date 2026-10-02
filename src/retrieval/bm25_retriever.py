"""BM25 检索器：基于 rank_bm25 + jieba 分词，兼容 .invoke() 接口。

设计要点：
- 从 data/processed/chunks.json 加载离线语料（与 Milvus 共享同一套 chunk_id）
- jieba 分词 + 教务术语自定义词典，避免专业术语被切碎
- 返回 list[Document]，metadata 带 source_file，评估代码零改动
"""
import json
from pathlib import Path

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# 教务高频专业术语，加入 jieba 自定义词典避免被切碎
DEFAULT_CUSTOM_TERMS = [
    "平均学分绩点",
    "综合测评",
    "创新学分",
    "勤工助学",
    "省政府奖学金",
    "限选",
    "通识选修",
    "专业必修",
    "专业选修",
    "公共基础",
    "培养方案",
    "毕业要求",
    "学分要求",
]


class BM25Retriever:
    """基于 rank_bm25 + jieba 的关键词检索器，兼容 .invoke() 接口。

    用法：
        retriever = BM25Retriever()
        docs = retriever.invoke("计算机专业毕业需要多少学分")
    """

    def __init__(
        self,
        chunks_path: str = "data/processed/chunks.json",
        top_k: int = 10,
        custom_terms: list[str] | None = None,
    ):
        self.top_k = top_k

        # 加载离线 chunk 语料
        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(
                f"chunk 语料不存在: {chunks_path}\n"
                "请先运行 build_vector_store() 导出 chunks.json"
            )
        with open(path, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        # 注册教务术语到 jieba 自定义词典
        terms = custom_terms if custom_terms is not None else DEFAULT_CUSTOM_TERMS
        for term in terms:
            jieba.add_word(term)

        # 建 BM25 索引：对每个 chunk 的 text 分词
        print(f"🔧 正在构建 BM25 索引（{len(self.chunks)} 条语料）...")
        tokenized_corpus = [list(jieba.cut(c["text"])) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 索引构建完成")

    def invoke(self, query: str) -> list[Document]:
        """检索与 query 关键词最匹配的 top_k 个 chunk。"""
        # query 用同一套分词逻辑
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        # 取分数最高的 top_k 个下标
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[: self.top_k]

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                # 分数为 0 说明无词重合，跳过
                continue
            c = self.chunks[i]
            results.append(
                Document(
                    page_content=c["text"],
                    metadata={
                        "source_file": c["source_file"],
                        "chunk_id": c.get("chunk_id", ""),
                        "score": float(scores[i]),
                    },
                )
            )
        return results
