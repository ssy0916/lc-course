import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# .env 在项目根目录（src 的上一级），需先于路由模块加载
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))

from common.logger import setup_logging, logger

# 初始化日志配置
setup_logging()

app = FastAPI(
    title="lc-course Recipe Agent API",
    description="A LangGraph recipe agent with a cited vector knowledge base.",
    version="0.2.0",
)

# 1. 配置跨域资源共享 (CORS)
# 插件开发中，由于请求来自浏览器扩展环境，必须正确配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定插件的 ID 或具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 挂载路由
from api.v1 import chat, knowledge  # noqa: E402

app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
app.include_router(knowledge.router, prefix="/api/v1", tags=["知识库"])

# OSS 路由在导入期就会校验凭证，.env 未配置 OSS 密钥时跳过挂载而不是崩溃
try:
    from api.v1 import oss  # noqa: E402

    app.include_router(oss.router, prefix="/api/v1", tags=["申请上传签名url"])
except Exception as e:
    logger.warning(f"oss 路由未挂载，请先在 .env 中配置 OSS 凭证: {e}")

# 3. 挂载前端资源
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# 前端 fallback 路由 - 只处理非 API 请求
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str):
    # 排除 API 路径
    if path.startswith("api/"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Not Found"}, status_code=404)
    # 如果请求的是静态文件，直接返回
    file_path = os.path.join(static_dir, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    # 否则返回 index.html (SPA fallback)
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "你的独家私厨上线了~", "status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # 启动命令：在 src 目录下运行 python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
