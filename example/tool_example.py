import os
from typing import List, Dict, Any, Union
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 设置模型参数
default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
default_model = "glm-4.5-flash"
default_api_key = os.environ.get("zhipuai_apikey")

# 初始化模型
model = ChatOpenAI(
    model=default_model,
    base_url=default_base_url,
    api_key=SecretStr(default_api_key) if default_api_key else None,
    temperature=0.5,
)


# 定义简单的工具
@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气信息"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴朗，温度15-25°C",
        "上海": "多云，温度20-28°C",
        "广州": "小雨，温度25-32°C",
        "深圳": "晴天，温度26-33°C",
    }
    return weather_data.get(location, f"未找到{location}的天气信息")


@tool
def calculate(a: float, b: float, operation: str) -> Union[float, str]:
    """执行简单的数学计算
    
    参数:
    a: 第一个数字
    b: 第二个数字
    operation: 操作类型，可以是add, subtract, multiply, divide
    
    返回:
    float: 计算结果
    str: 错误信息（当输入无效时）
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "除数不能为零"
        return a / b
    else:
        return f"不支持的操作: {operation}"


# 示例1: 基本工具调用
def basic_tool_call():
    """基本的工具调用示例"""
    # 为模型绑定工具
    model_with_tools = model.bind_tools([get_weather])
    
    # 调用模型
    response = model_with_tools.invoke("北京今天的天气怎么样？")
    
    print("基本工具调用结果:")
    print(f"响应对象: {response}")
    print(f"工具调用: {response.tool_calls}")
    
    # 如果模型决定使用工具，执行工具调用
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "get_weather":
            location = tool_call["args"]["location"]
            result = get_weather.invoke(location)
            print(f"工具执行结果: {result}")


# 示例2: 多工具调用
def multi_tool_call():
    """多工具调用示例"""
    # 为模型绑定多个工具
    model_with_tools = model.bind_tools([get_weather, calculate])
    
    # 调用模型
    response1 = model_with_tools.invoke("上海的天气如何？")
    response2 = model_with_tools.invoke("100加上200等于多少？")
    
    print("\n多工具调用结果:")
    
    # 处理第一个响应
    if response1.tool_calls:
        tool_call = response1.tool_calls[0]
        if tool_call["name"] == "get_weather":
            location = tool_call["args"]["location"]
            result = get_weather.invoke(location)
            print(f"上海天气: {result}")
    
    # 处理第二个响应
    if response2.tool_calls:
        tool_call = response2.tool_calls[0]
        if tool_call["name"] == "calculate":
            a = tool_call["args"]["a"]
            b = tool_call["args"]["b"]
            operation = tool_call["args"]["operation"]
            result = calculate.invoke({"a": a, "b": b, "operation": operation})
            print(f"计算结果: {result}")


# 示例3: 工具调用链
def tool_chain():
    """工具调用链示例"""
    # 创建一个包含工具调用的链
    prompt = ChatPromptTemplate.from_template("分析用户问题并调用适当的工具: {question}")
    
    # 定义处理工具调用的函数
    def handle_tool_call(model_response):
        if model_response.tool_calls:
            tool_call = model_response.tool_calls[0]
            if tool_call["name"] == "get_weather":
                return get_weather.invoke(tool_call["args"]["location"])
            elif tool_call["name"] == "calculate":
                return calculate.invoke(tool_call["args"])
        return model_response.content
    
    # 创建链
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | model.bind_tools([get_weather, calculate])
        | handle_tool_call
    )
    
    # 运行链
    result1 = chain.invoke("广州的天气怎么样？")
    result2 = chain.invoke("300乘以5等于多少？")
    
    print("\n工具调用链结果:")
    print(f"广州天气查询结果: {result1}")
    print(f"乘法计算结果: {result2}")


# 示例4: 自动工具调用
def auto_tool_call():
    """自动工具调用示例"""
    try:
        from langchain_core.runnables import RunnableLambda
        
        # 创建一个自动处理工具调用的函数
        def auto_tool_handler(input_data):
            response = model_with_tools.invoke(input_data)
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call["name"] == "get_weather":
                    return get_weather.invoke(tool_call["args"]["location"])
                elif tool_call["name"] == "calculate":
                    return calculate.invoke(tool_call["args"])
            return response.content
        
        # 为模型绑定工具
        model_with_tools = model.bind_tools([get_weather, calculate])
        
        # 创建自动工具调用链
        auto_chain = RunnableLambda(auto_tool_handler)
        
        # 运行链
        result = auto_chain.invoke("深圳的天气如何？")
        
        print("\n自动工具调用结果:")
        print(f"深圳天气: {result}")
    except ImportError:
        print("\n自动工具调用示例需要langchain>=0.1.10版本")


# 示例5: 批处理工具调用
def batch_tool_call():
    """批处理工具调用示例"""
    # 为模型绑定工具
    model_with_tools = model.bind_tools([get_weather, calculate])
    
    # 准备批处理输入
    inputs = [
        "北京的天气怎么样？",
        "200除以4等于多少？"
    ]
    
    # 执行批处理调用
    responses = model_with_tools.batch(inputs)
    
    print("\n批处理工具调用结果:")
    
    # 处理每个响应
    for i, response in enumerate(responses):
        print(f"查询 {i+1}:")
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call["name"] == "get_weather":
                location = tool_call["args"]["location"]
                result = get_weather.invoke(location)
                print(f"  天气结果: {result}")
            elif tool_call["name"] == "calculate":
                result = calculate.invoke(tool_call["args"])
                print(f"  计算结果: {result}")


if __name__ == "__main__":
    print("===== LangChain 工具调用示例 =====")
    
    # 运行各个示例
    basic_tool_call()
    multi_tool_call()
    tool_chain()
    auto_tool_call()
    batch_tool_call()
    
    print("\n所有示例运行完毕！")