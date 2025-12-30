import asyncio

from .mcp_client import client

DEFAULT_MCP_TOOLS = tuple(asyncio.run(mcp_client.get_mcp_tools()))
