import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from services.knowledge_base import knowledge_store

logger = logging.getLogger("personal_chef")

_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

# 注意：位置参数名是 model（写成 model_name 会得到不可直接使用的 _ConfigurableModel）
model = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    temperature=0.2,
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

web_search = TavilySearch(max_results=5, topic="general")


def build_checkpointer() -> SqliteSaver:
    resources_dir = _project_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        resources_dir / "personal_chef.db", check_same_thread=False
    )
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


checkpointer = build_checkpointer()
system_prompt = """
你是一名私人厨师。先根据用户的食材清单或图片理解需求，再按需调用工具。
涉及菜谱知识、食材搭配或烹饪方法时，优先查询 search_recipe_knowledge；
知识库无相关内容或需要最新信息时，可调用 tavily_search 联网补充。
使用知识库信息时，必须在对应结论后写出知识库返回的 [来源: ...] 标记。
严格区分知识库内容与联网搜索结果，不要伪造来源、营养数据或图片链接。
给出清晰、可执行的步骤；如果检索结果不足以回答，应明确说明。
"""

@tool
def search_recipe_knowledge(query: str) -> str:
    """Search the recipe and cooking knowledge base for grounded answers."""
    matches = knowledge_store.search(query)
    if not matches:
        return "知识库中没有找到相关内容；不要虚构知识库来源，可以继续使用联网搜索。"
    return "\n\n".join(
        f"[来源: {match['source']}]\n{match['content']}" for match in matches
    )


tools = [search_recipe_knowledge, web_search]

# LangGraph Dev manages its own checkpointing for the exported graph.
agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)

# The FastAPI service uses a local SQLite checkpointer for its chat history API.
local_agent = create_agent(
    model=model, tools=tools, checkpointer=checkpointer, system_prompt=system_prompt
)


def search_recipes(prompt: str, image: str | None, thread_id: str):
    logger.info("Recipe chat request received (thread_id=%s)", thread_id)
    if not image or not image.strip():
        message = HumanMessage(content=prompt)
    else:
        message = HumanMessage(
            content=[
                {"type": "image", "url": image},
                {"type": "text", "text": prompt},
            ]
        )

    for chunk, _ in local_agent.stream(
        input={"messages": [message]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            yield chunk.content


# 清空会话
def clear_messages(thread_id: str) -> None:
    logger.info("Clearing chat history (thread_id=%s)", thread_id)
    checkpointer.delete_thread(thread_id)


# 查询会话历史
def get_messages(thread_id: str) -> list[dict[str, str | list]]:
    logger.info("Loading chat history (thread_id=%s)", thread_id)
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if not checkpoint:
        return []

    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    result = []
    for msg in messages:
        if not msg.content:
            continue

        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result
def main() -> None:
    prompt = input("请输入食材清单: ").strip()
    for chunk in search_recipes(prompt, None, thread_id="cli-demo"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
