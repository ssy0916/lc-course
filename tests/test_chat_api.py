import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["DASHSCOPE_API_KEY"] = "test-api-key"
os.environ["DASHSCOPE_BASE_URL"] = "https://example.invalid/v1"
os.environ["TAVILY_API_KEY"] = "test-tavily-key"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from api.v1 import chat
from api.v1 import knowledge
from agents import personal_chef
from main import app


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_chat_stream_returns_sse_tokens_and_completion(self) -> None:
        with patch.object(
            chat, "search_recipes", return_value=iter(["第一段", "第二段"])
        ) as search:
            response = self.client.post(
                "/api/v1/chat/stream",
                json={
                    "message": "鸡蛋怎么做？",
                    "image_url": None,
                    "thread_id": "api-test",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"content": "第一段"', response.text)
        self.assertIn('"content": "第二段"', response.text)
        self.assertIn('"type": "done"', response.text)
        search.assert_called_once_with("鸡蛋怎么做？", None, "api-test")

    def test_chat_endpoint_rejects_empty_message(self) -> None:
        response = self.client.post(
            "/api/v1/chat/stream",
            json={"message": "   ", "thread_id": "api-test"},
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_stream_reports_agent_errors(self) -> None:
        def broken_stream():
            yield "部分回答"
            raise RuntimeError("provider unavailable")

        with patch.object(chat, "search_recipes", return_value=broken_stream()):
            response = self.client.post(
                "/api/v1/chat/stream",
                json={"message": "你好", "thread_id": "api-test"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"content": "部分回答"', response.text)
        self.assertIn('"type": "error"', response.text)
        self.assertNotIn('"type": "done"', response.text)

    def test_history_endpoint_returns_messages_for_thread(self) -> None:
        history = [{"role": "assistant", "content": "你好"}]
        with patch.object(chat, "get_messages", return_value=history):
            response = self.client.get(
                "/api/v1/chat/messages", params={"thread_id": "api-test"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"messages": history})

    def test_knowledge_endpoint_ingests_document(self) -> None:
        with patch.object(knowledge.knowledge_store, "ingest", return_value=2) as ingest:
            response = self.client.post(
                "/api/v1/knowledge/documents",
                json={"source": "recipes.md", "content": "番茄炒蛋做法"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"source": "recipes.md", "chunks_indexed": 2}
        )
        ingest.assert_called_once_with("recipes.md", "番茄炒蛋做法")

    def test_agent_knowledge_tool_preserves_citation_source(self) -> None:
        matches = [
            {"source": "recipes.md", "content": "番茄炒蛋做法", "score": 0.9}
        ]
        with patch.object(
            personal_chef.knowledge_store, "search", return_value=matches
        ):
            result = personal_chef.search_recipe_knowledge.invoke(
                {"query": "番茄炒蛋"}
            )

        self.assertIn("[来源: recipes.md]", result)
        self.assertIn("番茄炒蛋做法", result)

    def test_homepage_serves_the_chat_client(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("食材驱动的菜谱 Agent", response.text)


if __name__ == "__main__":
    unittest.main()
