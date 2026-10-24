import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from common.logger import logger
from models.schemas import ChatRequest
from agents.personal_chef import clear_messages, get_messages, search_recipes

router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    def stream_events():
        try:
            for content in search_recipes(
                request.message, request.image_url, request.thread_id
            ):
                event = {"type": "token", "content": content}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception:
            logger.exception("Chat stream failed (thread_id=%s)", request.thread_id)
            error = {"type": "error", "message": "生成失败，请检查模型配置或服务端日志。"}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    messages = get_messages(thread_id)
    return {"messages": messages}


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    clear_messages(thread_id)
    return {"success": True}