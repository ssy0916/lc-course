import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

_project_root = Path(__file__).resolve().parents[2]
_default_database = _project_root / "resources" / "recipe_knowledge.db"
_default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Stored and query embedding dimensions do not match")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


class KnowledgeStore:
    def __init__(
        self,
        database_path: Path = _default_database,
        embeddings: EmbeddingClient | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self._embeddings = embeddings

    def _embedding_client(self) -> EmbeddingClient:
        if self._embeddings is None:
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Set DASHSCOPE_API_KEY before importing or searching knowledge documents."
                )
            self._embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
                api_key=api_key,
                base_url=os.getenv("DASHSCOPE_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or _default_base_url,
            )
        return self._embeddings

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                UNIQUE(source, chunk_index)
            )
            """
        )
        return connection

    def ingest(self, source: str, content: str) -> int:
        chunks = split_text(content)
        if not chunks:
            raise ValueError("Document content must contain non-whitespace text")

        vectors = self._embedding_client().embed_documents(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned an unexpected number of vectors")

        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM knowledge_chunks WHERE source = ?", (source,))
            connection.executemany(
                """
                INSERT INTO knowledge_chunks (source, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (source, index, chunk, json.dumps(vector))
                    for index, (chunk, vector) in enumerate(
                        zip(chunks, vectors, strict=True)
                    )
                ],
            )
        return len(chunks)

    def search(self, query: str, limit: int = 4) -> list[dict[str, str | float]]:
        if not query.strip() or limit <= 0:
            return []

        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT source, content, embedding FROM knowledge_chunks"
            ).fetchall()
        if not rows:
            return []

        query_vector = self._embedding_client().embed_query(query)
        matches = [
            {
                "source": source,
                "content": content,
                "score": _cosine_similarity(json.loads(vector), query_vector),
            }
            for source, content, vector in rows
        ]
        return sorted(matches, key=lambda match: float(match["score"]), reverse=True)[
            :limit
        ]


knowledge_store = KnowledgeStore()
