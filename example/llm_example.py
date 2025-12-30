import os

from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

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


def simple_msg():
	response = model.invoke("介绍一下牛顿?")
	print(response)
	print("结果：", response.content)


def multi_msg():
	# 中文仿写：多轮对话（中文翻译成英文）
	conversation = [
		{"role": "system", "content": "你是一个专业的翻译助手，负责将中文准确翻译成英文，翻译结果要简洁自然。"},
		{"role": "user", "content": "翻译：我喜欢编程。"},
		{"role": "assistant", "content": "I love programming."},
		{"role": "user", "content": "翻译：我喜欢开发应用程序。"}
	]
	# 调用模型获取响应
	response = model.invoke(conversation)
	print(response)
	# 打印结果（提取 AIMessage 中的内容）
	print("结果：", response.content)


def multi_obj_msg():
	conversation = [
		# 系统消息：定义助手角色（中文翻译助手）
		SystemMessage(content="你是专业的翻译助手，负责将中文准确翻译成自然流畅的英文，无需额外解释，仅返回翻译结果。"),
		# 用户消息：第一轮翻译请求
		HumanMessage(content="翻译：我喜欢编程。"),
		# 助手历史回复：第一轮翻译结果
		AIMessage(content="I love programming."),
		# 用户消息：第二轮翻译请求（延续上下文）
		HumanMessage(content="翻译：我喜欢开发应用程序。")
	]

	# 调用模型获取响应（返回 AIMessage 对象）
	response = model.invoke(conversation)

	# 打印结果
	print("结果：", response)
	print("翻译结果（提取内容）：", response.content)


def stream_msg():
	for chunk in model.stream("鹦鹉为什么有彩色的羽毛？"):
		print(chunk.text, end='', flush=True)


def batch_msg():
	responses = model.batch(inputs=[
		"鹦鹉为什么有彩色的羽毛？",
		"飞机是如何飞行的？",
		"什么是量子计算？"
	], config=RunnableConfig(max_concurrency=2))
	for response in responses:
		print(response)


if __name__ == '__main__':
	# simple_msg()
	# multi_msg()
	# multi_obj_msg()
	# stream_msg ()
	# batch_msg()
	pass