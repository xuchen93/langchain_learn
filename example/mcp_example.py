import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

# MCP配置
host = "172.16.116.170"
port = 8000
MCP_SERVER_URL = f"http://{host}:{port}/mcp"

# 创建LLM模型实例（与项目保持一致）
_base_url = "https://open.bigmodel.cn/api/paas/v4/"
_model = "glm-4.5-flash"
_api_key = os.environ.get("zhipuai_apikey")

model = ChatOpenAI(
    model=_model,
    base_url=_base_url,
    api_key=_api_key,
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 创建MCP客户端实例
def create_mcp_client():
    """创建MCP客户端实例"""
    # 直接使用实际的MCP客户端
    client = MultiServerMCPClient(
        {
            "default": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        }
    )
    print(f"初始化MCP客户端成功，连接到: {MCP_SERVER_URL}")
    return client

# 全局MCP客户端实例
mcp_client = create_mcp_client()


async def list_mcp_tools():
    """列出所有可用的MCP工具"""
    print("=== 获取MCP工具列表 ===")
    try:
        tools = await mcp_client.get_tools()
        print(f"找到 {len(tools)} 个MCP工具:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        return tools
    except Exception as e:
        print(f"获取MCP工具失败: {e}")
        return []


async def use_mcp_tool_directly():
    """直接使用MCP工具"""
    print("\n=== 直接使用MCP工具 ===")
    try:
        tools = await mcp_client.get_tools()
        if tools:
            # 查找search_active_task工具
            task_tool = next((t for t in tools if t.name == "search_active_task"), None)
            if task_tool:
                print("使用search_active_task工具...")
                # 尝试使用正确的方式调用工具
                result = await task_tool.ainvoke({})
                print(f"工具执行结果: {result}")
            else:
                print("未找到search_active_task工具")
    except Exception as e:
        print(f"使用MCP工具失败: {e}")


async def use_mcp_session():
    """使用MCP会话（有状态的工具调用）"""
    print("\n=== 使用MCP会话（有状态调用） ===")
    try:
        # 简化会话处理逻辑
        print("演示MCP工具的异步调用...")
        # 直接使用客户端获取工具
        tools = await mcp_client.get_tools()
        print(f"通过客户端获取到 {len(tools)} 个工具")
        if tools:
            task_tool = next((t for t in tools if t.name == "search_active_task"), None)
            if task_tool:
                print("使用search_active_task工具（异步调用）...")
                # 使用异步方法调用工具
                result = await task_tool.ainvoke({})
                print(f"工具执行结果: {result}")
        print("MCP工具异步调用演示完成")
    except Exception as e:
        print(f"演示MCP工具异步调用时出错: {e}")


async def use_multiple_mcp_servers():
    """使用多个MCP服务器（示例）"""
    print("\n=== 配置多个MCP服务器（示例） ===")
    try:
        # 这个函数展示了如何配置多个MCP服务器
        # 实际使用时需要替换为真实的服务器地址
        multi_server_client = MultiServerMCPClient(
            {
                "default": {
                    "transport": "streamable_http",
                    "url": MCP_SERVER_URL,
                },
                "another_server": {
                    "transport": "streamable_http",
                    "url": "http://example.com/mcp",  # 示例地址
                }
            }
        )
        print("配置多服务器MCP客户端成功")
    except Exception as e:
        print(f"配置多服务器MCP客户端失败: {e}")


async def run_all_examples():
    """运行所有示例"""
    # 列出MCP工具
    await list_mcp_tools()
    
    # 直接使用MCP工具
    await use_mcp_tool_directly()
    
    # 使用MCP会话（有状态的工具调用）
    await use_mcp_session()
    
    # 配置多个MCP服务器（示例）
    await use_multiple_mcp_servers()


if __name__ == "__main__":
    # 可以通过注释控制要运行的示例
    # asyncio.run(list_mcp_tools())
    # asyncio.run(use_mcp_tool_directly())
    # asyncio.run(use_mcp_session())
    # asyncio.run(use_multiple_mcp_servers())
    asyncio.run(run_all_examples())