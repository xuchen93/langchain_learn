from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import get_logger, settings
from core.mcp import DEFAULT_MCP_TOOLS
from core.middle import DEFAULT_MIDDLEWARE
from core.tools import DEFAULT_TOOLS

logger = get_logger(__name__)


def create_default_model():
	model = ChatOpenAI(
		model=settings.openai_model,
		base_url=settings.openai_api_url,
		api_key=settings.openai_api_key,
		temperature=settings.openai_temperature,
	)
	return model


def create_default_agent(middle_list=DEFAULT_MIDDLEWARE, tools=DEFAULT_TOOLS, mcp_tools=DEFAULT_MCP_TOOLS):
	model = create_default_model()
	active_tools = tools + mcp_tools
	return create_agent(model, tools=active_tools, middleware=middle_list)


if __name__ == '__main__':
	agent = create_default_agent()
	print(agent)
