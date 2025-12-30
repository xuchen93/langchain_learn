"""
时间相关工具
提供获取当前时间、日期等功能
"""

from datetime import datetime

from langchain_core.tools import tool

from config import get_logger

logger = get_logger(__name__)


@tool
def get_current_time() -> str:
	"""
	获取当前时间

	返回格式化的当前日期和时间，格式为：YYYY-MM-DD HH:MM:SS

	Returns:
		当前时间的字符串表示

	使用场景：
		- 当用户明确问"现在几点"、"当前时间"时使用
	"""
	current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	logger.debug(f"🕐 获取当前时间: {current_time}")
	return f"当前时间是：{current_time}"
