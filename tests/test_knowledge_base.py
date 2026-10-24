import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.knowledge_base import KnowledgeStore, split_text


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if "鸡蛋" in text:
            return [1.0, 0.0]
        if "牛肉" in text:
            return [0.0, 1.0]
        return [0.2, 0.2]


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = KnowledgeStore(
            Path(self.temp_dir.name) / "knowledge.db",
            embeddings=FakeEmbeddings(),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_split_text_creates_overlapping_chunks(self) -> None:
        chunks = split_text("abcdefghijklmnop", chunk_size=10, overlap=3)

        self.assertEqual(chunks, ["abcdefghij", "hijklmnop"])

    def test_ingest_and_search_returns_best_vector_match_and_source(self) -> None:
        self.store.ingest("eggs.md", "鸡蛋可以炒制或煮熟。")
        self.store.ingest("beef.md", "牛肉适合炖煮。")

        matches = self.store.search("鸡蛋怎么做？", limit=1)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["source"], "eggs.md")
        self.assertIn("鸡蛋", str(matches[0]["content"]))
        self.assertAlmostEqual(float(matches[0]["score"]), 1.0)

    def test_ingesting_same_source_replaces_existing_chunks(self) -> None:
        self.store.ingest("recipe.md", "鸡蛋炒番茄。")
        self.store.ingest("recipe.md", "鸡蛋蒸熟。")

        matches = self.store.search("鸡蛋", limit=10)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["content"], "鸡蛋蒸熟。")


if __name__ == "__main__":
    unittest.main()
