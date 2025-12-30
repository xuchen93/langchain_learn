"""
统一配置管理模块
使用 Pydantic Settings 管理所有配置项，支持从环境变量和 .env 文件加载
"""
import os
from typing import Optional, Dict, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	"""
	应用配置类

	所有配置项都可以通过环境变量或 .env 文件设置
	优先级：环境变量 > .env 文件 > 默认值
	"""

	# ==================== OpenAI 配置 ====================
	openai_api_key: str = Field(
		default=os.environ.get("zhipuai_apikey"),
		description="OpenAI API 密钥，必须设置"
	)

	openai_api_url: str = Field(
		default="https://open.bigmodel.cn/api/paas/v4/",
		description="OpenAI API 地址"
	)

	openai_model: str = Field(
		default="glm-4.5-flash",
		description="默认使用的 智谱AI 模型"
	)

	openai_temperature: float = Field(
		default=0.1,
		ge=0.0,
		le=2.0,
		description="模型温度参数，控制输出的随机性"
	)

	openai_max_tokens: Optional[int] = Field(
		default=None,
		description="最大生成 token 数，None 表示使用模型默认值"
	)

	openai_streaming: bool = Field(
		default=True,
		description="是否默认启用流式输出"
	)

	# ==================== 服务器配置 ====================
	server_host: str = Field(
		default="127.0.0.1",
		description="服务器监听地址"
	)

	server_port: int = Field(
		default=8000,
		ge=1,
		le=65535,
		description="服务器监听端口"
	)

	server_reload: bool = Field(
		default=True,
		description="开发模式下是否自动重载"
	)

	# ==================== 日志配置 ====================
	log_level: str = Field(
		default="INFO",
		description="日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL"
	)

	log_file: str = Field(
		default="logs/app.log",
		description="日志文件路径"
	)

	log_rotation: str = Field(
		default="10 MB",
		description="日志文件轮转大小"
	)

	log_retention: str = Field(
		default="3 days",
		description="日志文件保留时间"
	)

	# ==================== 应用配置 ====================
	app_name: str = Field(
		default="langchain_example",
		description="应用名称"
	)

	app_version: str = Field(
		default="0.0.1",
		description="应用版本"
	)

	debug: bool = Field(
		default=False,
		description="是否启用调试模式"
	)

	# ==================== MCP 服务配置 ====================
	# MCP 服务配置字典，用于支持多服务配置
	# 格式: {"service_name": {"transport": "streamable_http", "url": "http://localhost:12345/mcp"}, ...}
	mcp_services: Dict[str, Dict[str, Any]] = Field(
		default_factory=dict,
		description="MCP 服务配置字典，支持多个 MCP 服务"
	)

	# Pydantic Settings 配置
	model_config = SettingsConfigDict(
		env_file=".env",  # 从 .env 文件加载
		env_file_encoding="utf-8",
		case_sensitive=False,  # 环境变量不区分大小写
		extra="ignore",  # 忽略额外的环境变量
	)

	def get_openai_config(self) -> dict:
		"""
		获取 OpenAI 配置字典

		Returns:
			包含 OpenAI 配置的字典
		"""
		config = {
			"api_key": self.openai_api_key,
			"base_url": self.openai_api_url,
			"model": self.openai_model,
			"temperature": self.openai_temperature,
		}

		if self.openai_max_tokens is not None:
			config["max_tokens"] = self.openai_max_tokens

		return config

	def get_mcp_clients(self) -> Dict[str, Dict[str, Any]]:
		"""
		获取 MCP 服务配置字典

		Returns:
			包含所有 MCP 服务配置的字典
		"""
		mcp_clients = {
			"cbz-http": {
				"transport": "streamable_http",
				"url": "http://172.16.116.170:8000/mcp",
			}
		}

		# 如果用户提供了额外的 MCP 服务配置，合并它们
		if self.mcp_services:
			mcp_clients.update(self.mcp_services)

		return mcp_clients


# 创建全局配置实例
settings = Settings()
