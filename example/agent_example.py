import os
import datetime

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage

# 大模型配置（与llm_example.py保持一致）
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


# 定义工具
def define_tools():
    """定义智能体可以使用的工具"""
    
    @tool
    def search(query: str) -> str:
        """搜索信息。当需要查找最新或特定信息时使用此工具。"""
        # 实际应用中，这里应该调用真实的搜索引擎API
        return f"搜索结果：{query} 的相关信息已找到"
    
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式。当需要进行数学计算时使用此工具。"""
        try:
            # 简单的计算示例，实际应用中应使用更安全的计算方法
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    @tool
    def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """获取当前时间。当需要知道当前时间时使用此工具。"""
        current_time = datetime.datetime.now()
        return f"当前时间：{current_time.strftime(format)}"
    
    return [search, calculate, get_current_time]


# 创建基本智能体
def create_basic_agent():
    """创建基本的智能体示例"""
    print("=== 创建基本智能体 ===")
    tools = define_tools()
    agent = create_agent(model, tools=tools)
    return agent


# 使用智能体
def use_basic_agent():
    """使用基本智能体执行任务"""
    print("\n=== 使用基本智能体执行任务 ===")
    agent = create_basic_agent()
    
    try:
        # 执行需要搜索的任务
        result1 = agent.invoke({
            "messages": [HumanMessage(content="告诉我当前时间")]
        })
        print(f"搜索任务结果：{result1['messages'][-1].content}")
        
        # 执行需要计算的任务
        result2 = agent.invoke({
            "messages": [HumanMessage(content="计算123乘以456的结果")]
        })
        print(f"计算任务结果：{result2['messages'][-1].content}")
    except Exception as e:
        print(f"智能体执行失败：{e}")


# 动态模型选择
def create_dynamic_model_agent():
    """创建具有动态模型选择功能的智能体"""
    print("\n=== 创建具有动态模型选择功能的智能体 ===")
    
    # 定义不同的模型（在实际应用中，这些应该是不同能力或成本的模型）
    basic_model = ChatOpenAI(
        model=_model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=0.3,
    )
    
    advanced_model = ChatOpenAI(
        model=_model,  # 在实际应用中可以使用更高级的模型
        base_url=_base_url,
        api_key=_api_key,
        temperature=0.7,
    )
    
    # 创建动态模型选择中间件
    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        """根据对话复杂性选择模型"""
        if request.state and "messages" in request.state:
            message_count = len(request.state["messages"])
            if message_count > 3:
                # 对复杂或长对话使用高级模型
                updated_request = request.override(model=advanced_model)
                print("使用高级模型处理复杂对话")
            else:
                updated_request = request.override(model=basic_model)
                print("使用基础模型处理简单对话")
        else:
            updated_request = request
        
        return handler(updated_request)
    
    tools = define_tools()
    agent = create_agent(
        model=basic_model,  # 默认模型
        tools=tools,
        middleware=[dynamic_model_selection]
    )
    
    return agent


# 使用动态模型智能体
def use_dynamic_model_agent():
    """使用具有动态模型选择功能的智能体"""
    print("\n=== 使用具有动态模型选择功能的智能体 ===")
    agent = create_dynamic_model_agent()
    
    try:
        # 简单对话
        result1 = agent.invoke({
            "messages": [HumanMessage(content="我想查看当前的时间")]
        })
        print(f"简单对话结果：{result1['messages'][-1].content}")
        
        # 模拟复杂对话（通过构建消息历史）
        from langchain_core.messages import AIMessage
        messages = [
            HumanMessage(content="你好，我是一个用户"),
            AIMessage(content="你好！我是你的助手，有什么可以帮助你的吗？"),
            HumanMessage(content="我想知道一个小时之后的时间")
        ]
        
        result2 = agent.invoke({"messages": messages})
        print(f"复杂对话结果：{result2['messages'][-1].content}")
    except Exception as e:
        print(f"动态模型智能体执行失败：{e}")


# 工具错误处理
def create_tool_error_handling_agent():
    """创建具有自定义工具错误处理的智能体"""
    print("\n=== 创建具有自定义工具错误处理的智能体 ===")
    
    # 创建工具错误处理中间件
    @wrap_model_call
    def handle_tool_errors(request, handler):
        """自定义工具错误处理"""
        try:
            return handler(request)
        except Exception as e:
            # 向模型返回自定义错误消息
            if hasattr(request, 'tool_call') and request.tool_call:
                return ToolMessage(
                    content=f"工具错误：请检查您的输入并重试。({str(e)})",
                    tool_call_id=request.tool_call["id"]
                )
            raise
    
    tools = define_tools()
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[handle_tool_errors]
    )
    
    return agent


# 使用带错误处理的智能体
def use_error_handling_agent():
    """使用具有自定义工具错误处理的智能体"""
    print("\n=== 使用具有自定义工具错误处理的智能体 ===")
    agent = create_tool_error_handling_agent()
    
    try:
        # 发送可能导致工具错误的请求
        result = agent.invoke({
            "messages": [HumanMessage(content="计算1/0的结果")]
        })
        print(f"错误处理结果：{result['messages'][-1].content}")
    except Exception as e:
        print(f"错误处理智能体执行失败：{e}")


# 主函数
def main():
    """主函数，运行所有示例"""
    print("LangChain智能体示例")
    
    # 为了避免API请求过多，我们先只运行动态模型智能体示例
    # 使用基本智能体
    # use_basic_agent()
    
    # 使用动态模型智能体
    use_dynamic_model_agent()
    
    # 使用带错误处理的智能体
    # use_error_handling_agent()


if __name__ == "__main__":
    main()