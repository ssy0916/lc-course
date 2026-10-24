"""把 examples/recipes 下的示例文档导入本地知识库。

用法：
    uv run python scripts/seed_knowledge.py

需要先在 .env 中配置 DASHSCOPE_API_KEY（用于 Embeddings）。
同一 source 重复导入会替换旧内容，可以安全重复执行。
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from services.knowledge_base import knowledge_store


def main() -> None:
    recipes_dir = project_root / "examples" / "recipes"
    files = sorted(recipes_dir.glob("*.md"))
    if not files:
        print(f"未在 {recipes_dir} 找到示例文档")
        return

    for path in files:
        content = path.read_text(encoding="utf-8")
        chunks = knowledge_store.ingest(source=path.name, content=content)
        print(f"已导入 {path.name}（{chunks} 个切片）")

    print(f"\n完成，共导入 {len(files)} 篇文档。")


if __name__ == "__main__":
    main()
