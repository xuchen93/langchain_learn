import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routers import chat
from config import settings, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""
	应用生命周期管理

	在应用启动和关闭时执行必要的初始化和清理工作
	"""
	# ==================== 启动时 ====================
	logger.info("=" * 60)
	logger.info(f"{settings.app_name} starting...")
	logger.info("=" * 60)

	# 打印配置信息
	logger.info(f"📊 运行环境:")
	logger.info(f"   - 模型: {settings.openai_model}")
	logger.info(f"   - API Base: {settings.openai_api_url}")
	logger.info(f"   - 调试模式: {settings.debug}")
	logger.info(f"   - 日志级别: {settings.log_level}")
	if settings.get_mcp_clients():
		logger.info(f"   - mcp服务: {settings.get_mcp_clients().keys()}")

	logger.info("=" * 60)
	logger.info("started")
	if settings.debug:
		logger.info("APP CHAT URL：http://localhost:8000/chat/stream")
	logger.info("=" * 60)

	yield

	# ==================== 关闭时 ====================
	logger.info("=" * 60)
	logger.info("stop...")
	logger.info("=" * 60)


app = FastAPI(
	title=settings.app_name,
	lifespan=lifespan
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
	"""
	记录所有 HTTP 请求的日志

	包括：请求方法、路径、耗时、状态码
	"""
	start_time = time.time()

	# 记录请求
	logger.info(f"📥 {request.method} {request.url.path}")

	# 处理请求
	try:
		response = await call_next(request)

		# 计算耗时
		process_time = time.time() - start_time

		# 记录响应
		logger.info(
			f"📤 {request.method} {request.url.path} "
			f"- {response.status_code} - {process_time:.3f}s"
		)

		# 添加响应头
		response.headers["X-Process-Time"] = str(process_time)

		return response

	except Exception as e:
		process_time = time.time() - start_time
		logger.error(
			f"❌ {request.method} {request.url.path} "
			f"- 错误: {str(e)} - {process_time:.3f}s"
		)
		raise


# ==================== 异常处理 ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
	"""
	全局异常处理器

	捕获所有未处理的异常，返回统一的错误响应
	"""
	logger.error(f"❌ 未处理的异常: {exc}", exc_info=True)

	return JSONResponse(
		status_code=500,
		content={
			"error": "Internal Server Error",
			"message": str(exc) if settings.debug else "服务器内部错误",
			"path": str(request.url),
		},
	)


# ==================== 路由注册 ====================
# 注册聊天路由
app.include_router(chat.router)

# ==================== 开发服务器启动 ====================

if __name__ == "__main__":
	import uvicorn

	logger.info("🔧 以开发模式启动服务器...")
	uvicorn.run(
		"api.http_server:app",
		host=settings.server_host,
		port=settings.server_port,
		reload=settings.server_reload,
		log_level=settings.log_level.lower(),
	)
