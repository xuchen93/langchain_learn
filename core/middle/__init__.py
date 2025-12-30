"""
中间件
link:https://langchain-doc.cn/v1/python/langchain/middleware.html#%E7%BC%93%E5%AD%98-anthropic-prompt-caching
"""
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI

from config import settings

_DEFAULT_MODEL = ChatOpenAI(
	model=settings.openai_model,
	base_url=settings.openai_api_url,
	api_key=settings.openai_api_key,
	temperature=settings.openai_temperature,
)

DEFAULT_MIDDLEWARE = (
	# 超过上下文窗口的长期对话
	# 具有大量历史记录的多轮对话
	# 需要保留完整对话上下文的应用
	SummarizationMiddleware(
		model=_DEFAULT_MODEL,
		max_tokens_before_summary=4000,
		messages_to_keep=20
	),
	# 防止失控代理进行过多 API 调用
	# 在生产部署中强制执行成本控制
	# 在特定调用预算内测试代理行为
	ModelCallLimitMiddleware(
		thread_limit=10,  # 每个线程（跨多次运行）最多 10 次调用
		run_limit=10,  # 每次运行（单次调用）最多 5 次调用
		exit_behavior="end",  # 或者 "error" 以引发异常
	),
	# 防止对昂贵的外部 API 进行过多调用
	# 限制网页搜索或数据库查询
	# 对特定工具使用强制执行速率限制
	ToolCallLimitMiddleware(
		thread_limit=10,
		run_limit=10
	),
)
