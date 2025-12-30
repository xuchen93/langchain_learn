import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# 配置
_base_url = "https://open.bigmodel.cn/api/paas/v4/"
_model = "glm-4.5-flash"
_api_key = os.environ.get("zhipuai_apikey")

# MCP服务器配置
host = "172.16.116.170"
port = 8000

# 创建LLM模型实例
model = ChatOpenAI(
	model=_model,
	base_url=_base_url,
	api_key=_api_key,
	temperature=0.5,
	max_tokens=None,
	timeout=None,
	max_retries=2,
)

# 创建多MCP服务器客户端实例
_mcp_client = MultiServerMCPClient(
	{
		"default": {
			"transport": "streamable_http",
			"url": f"http://172.16.116.170:8000/mcp",
		}
		# 可以添加更多MCP服务器配置
		# "math": {
		#     "transport": "stdio",
		#     "command": "python",
		#     "args": ["/path/to/math_server.py"],
		# }
	}
)


async def get_mcp_tools():
	"""获取MCP工具列表"""
	tools = await _mcp_client.get_tools()
	print(f"获取到 {len(tools)} 个MCP工具")
	for tool in tools:
		print(f"工具名称: {tool.name}, 描述: {tool.description}")
	return tools


async def direct_tool_invoke():
	"""直接调用MCP工具查询租户任务并发数"""
	print("\n=== 直接调用MCP工具查询 ===")
	tools = await get_mcp_tools()
	if not tools:
		return "未找到可用的MCP工具"

	try:
		# 使用第一个工具查询数据
		result = await tools[0].ainvoke({})
		print(f"工具调用结果: {result}")
		return result
	except Exception as e:
		print(f"工具调用失败: {e}")
		return str(e)


async def as_stream_updates():
	"""创建智能体并查询租户任务并发数"""
	print("\n=== 创建智能体并查询 ===")
	tools = await get_mcp_tools()
	if not tools:
		return "未找到可用的MCP工具"

	try:
		# 创建智能体
		agent = create_agent(model, tools)
		async for chunk in agent.astream(  # [!code highlight]
				{"messages": [{"role": "user", "content": "帮我查询一下租户4092的各个任务并发数"}]},
				stream_mode="updates",
		):
			for step, data in chunk.items():
				print(f"step: {step}")
				print(f"content: {data['messages'][-1].content_blocks}")
	except Exception as e:
		print(f"智能体查询失败: {e}")
		return str(e)


async def as_stream_message():
	"""创建智能体并查询租户任务并发数"""
	print("\n=== 创建智能体并查询 ===")
	tools = await get_mcp_tools()
	if not tools:
		return "未找到可用的MCP工具"

	try:
		langgraph_step = None
		# 创建智能体
		agent = create_agent(model, tools)
		async for token, metadata in agent.astream(  # [!code highlight]
				{"messages": [{"role": "user", "content": "帮我查询一下租户4092的各个任务并发数"}]},
				stream_mode="messages",
		):
			if metadata["langgraph_step"] != langgraph_step:
				langgraph_step = metadata["langgraph_step"]
				print(f'\n当前执行第{langgraph_step}步，langgraph_node={metadata["langgraph_node"]}')
			if metadata['langgraph_node'] != "tools" and token.content:
				print(token.content, end='')
	except Exception as e:
		print(f"智能体查询失败: {e}")
		return str(e)


if __name__ == '__main__':
	print("=== LangChain MCP智能体查询示例 ===")
	# asyncio.run(direct_tool_invoke())
	# asyncio.run(as_stream_query())
	asyncio.run(as_stream_message())
	pass
