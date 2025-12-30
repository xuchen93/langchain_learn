import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import base_agent
from config import get_logger

logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
	"""聊天请求模型"""
	message: str = Field(..., description="用户消息", min_length=1)
	user_id: str = Field(..., description="用户ID")


@router.post("/stream/default")
async def chat(request: ChatRequest):
	async def generate():
		agent = base_agent.create_default_agent()
		langgraph_step = None
		async for token, metadata in agent.astream(  # [!code highlight]
				{"messages": [{"role": "user", "content": request.message}]},
				stream_mode="messages",
		):
			if metadata["langgraph_step"] != langgraph_step:
				langgraph_step = metadata["langgraph_step"]
				logger.info(f'\n当前执行第{langgraph_step}步，langgraph_node={metadata["langgraph_node"]}')
			if metadata['langgraph_node'] != "tools" and token.content:
				logger.info(token.content, end='')
				yield token.content

	return StreamingResponse(
		generate(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
		},
	)


@router.post("/stream/example")
async def chat_stream_example():
	"""
	流式聊天接口（SSE - Server-Sent Events）- 增强版

	接收用户消息，以流式方式返回 AI 的回复。
	适合需要实时显示生成过程的场景。

	增强功能:
	- 支持工具调用详情输出
	- 支持推理过程输出
	- 支持 token 使用统计
	- 支持来源引用输出
	- 支持计划和任务输出

	Args:
		request: 聊天请求

	Returns:
		SSE 流式响应

	响应格式（SSE）:
		```
		data: {"type": "start", "message": "开始生成..."}
		data: {"type": "chunk", "content": "文本内容"}
		data: {"type": "tool", "data": {...}}
		data: {"type": "reasoning", "data": {...}}
		data: {"type": "source", "data": {...}}
		data: {"type": "context", "data": {...}}
		data: {"type": "end", "message": "生成完成"}
		```
	"""

	# logger.info(f"🌊 收到流式聊天请求: {request.message[:50]}...")

	async def generate():
		for i in range(1, 10):
			time.sleep(0.1)
			yield f"current:{i}\n"

	# 返回 SSE 响应
	return StreamingResponse(
		generate(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
		},
	)
