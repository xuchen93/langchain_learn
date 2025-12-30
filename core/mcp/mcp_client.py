from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import get_logger, settings

logger = get_logger(__name__)

client = MultiServerMCPClient(settings.get_mcp_clients())


def get_mcp_clients() -> List[str]:
	return settings.get_mcp_clients().keys()


async def get_mcp_tools() -> list[BaseTool]:
	return await client.get_tools()
