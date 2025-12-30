import os
import json
from typing import List, Dict, Any, Union, Optional
from pydantic import SecretStr, BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

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


# 定义Pydantic模型用于结构化输出
class WeatherInfo(BaseModel):
    """天气信息模型"""
    location: str
    temperature: str
    condition: str
    suggestion: str


class ProductInfo(BaseModel):
    """产品信息模型"""
    name: str
    price: float
    description: str
    category: str


class BookInfo(BaseModel):
    """书籍信息模型"""
    title: str
    author: str
    publication_year: int
    genre: str
    rating: float


# 通用的结构化输出函数，带重试和错误处理
# 将参数类型从BaseModel改为type[BaseModel]
def get_structured_output(prompt: str, model_class: type[BaseModel]) -> Optional[BaseModel]:
    """获取结构化输出的通用函数，带重试和错误处理"""
    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=model_class)
    
    # 创建包含格式指令的提示词
    prompt_template = ChatPromptTemplate.from_template("""
    请严格按照以下JSON格式返回数据，不要添加任何额外的解释或说明文字：
    {format_instructions}
    
    {query}
    """)
    
    # 创建链
    chain = (
        prompt_template
        | model
        | RunnablePassthrough()
    )
    
    try:
        # 运行链
        result = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": prompt
        })
        
        # 提取内容
        content = result.content.strip()
        
        # 尝试直接解析JSON
        try:
            data = json.loads(content)
            return model_class(**data)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            # 查找JSON开始和结束位置
            start_pos = content.find('{')
            end_pos = content.rfind('}') + 1
            
            if start_pos != -1 and end_pos != -1:
                json_str = content[start_pos:end_pos]
                try:
                    data = json.loads(json_str)
                    return model_class(**data)
                except (json.JSONDecodeError, ValidationError):
                    print(f"解析提取的JSON失败: {json_str}")
            
            # 如果都失败，返回None
            print(f"无法解析响应为JSON: {content}")
            return None
    except Exception as e:
        print(f"获取结构化输出时出错: {e}")
        return None


# 示例1: 基本结构化输出
def basic_structured_output():
    """基本的结构化输出示例"""
    print("\n===== 示例1: 基本结构化输出 =====")
    
    # 使用自定义函数获取结构化输出
    response = get_structured_output("请提供北京今天的天气信息", WeatherInfo)
    
    if response:
        print("结构化响应:")
        print(f"类型: {type(response)}")
        print(f"位置: {response.location}")
        print(f"温度: {response.temperature}")
        print(f"天气状况: {response.condition}")
        print(f"建议: {response.suggestion}")
    else:
        print("未能获取有效的结构化输出")


# 示例2: 使用bind_tools进行结构化输出
def structured_output_with_bind_tools():
    """使用bind_tools进行结构化输出示例"""
    print("\n===== 示例2: 使用bind_tools进行结构化输出 =====")
    
    try:
        # 使用bind_tools获取结构化输出
        structured_model = model.bind_tools([WeatherInfo], tool_choice="WeatherInfo")
        
        # 调用模型
        response = structured_model.invoke("请提供上海今天的天气信息")
        
        print("响应对象:")
        
        # 提取结构化数据
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            weather_data = tool_call["args"]
            print(f"结构化数据: {weather_data}")
            # 可以转换为Pydantic模型
            weather_info = WeatherInfo(**weather_data)
            print(f"转换为Pydantic模型: {weather_info}")
        else:
            print(f"模型未返回工具调用: {response.content}")
    except Exception as e:
        print(f"使用bind_tools时出错: {e}")


# 示例3: 多类型结构化输出
def multi_type_structured_output():
    """多类型结构化输出示例"""
    print("\n===== 示例3: 多类型结构化输出 =====")
    
    # 测试不同类型的结构化输出
    weather_result = get_structured_output("请提供广州今天的天气信息", WeatherInfo)
    product_result = get_structured_output("请提供一款智能手机的信息", ProductInfo)
    book_result = get_structured_output("请提供一本经典小说的信息", BookInfo)
    
    if weather_result:
        print("天气信息:")
        print(f"  位置: {weather_result.location}")
        print(f"  温度: {weather_result.temperature}")
    
    if product_result:
        print("产品信息:")
        print(f"  名称: {product_result.name}")
        print(f"  价格: {product_result.price}")
    
    if book_result:
        print("书籍信息:")
        print(f"  标题: {book_result.title}")
        print(f"  作者: {book_result.author}")


# 示例4: 结构化输出链
def structured_output_chain():
    """结构化输出链示例"""
    print("\n===== 示例4: 结构化输出链 =====")
    
    # 创建一个包含结构化输出的链
    parser = PydanticOutputParser(pydantic_object=WeatherInfo)
    prompt = ChatPromptTemplate.from_template("""
    请严格按照以下JSON格式返回数据，不要添加任何额外的解释或说明文字：
    {format_instructions}
    
    请提供{location}今天的天气信息
    """)
    
    # 创建链
    chain = (
        {"location": RunnablePassthrough(), "format_instructions": lambda _: parser.get_format_instructions()}
        | prompt
        | model
        | RunnablePassthrough()
    )
    
    # 运行链
    try:
        result = chain.invoke("深圳")
        
        # 尝试解析结果
        content = result.content.strip()
        try:
            # 尝试提取JSON部分
            start_pos = content.find('{')
            end_pos = content.rfind('}') + 1
            
            if start_pos != -1 and end_pos != -1:
                json_str = content[start_pos:end_pos]
                data = json.loads(json_str)
                weather_info = WeatherInfo(**data)
                
                print("链的输出结果:")
                print(f"  位置: {weather_info.location}")
                print(f"  温度: {weather_info.temperature}")
                print(f"  天气状况: {weather_info.condition}")
        except Exception as e:
            print(f"解析链结果时出错: {e}")
            print(f"原始结果: {content}")
    except Exception as e:
        print(f"运行链时出错: {e}")


# 示例5: 批量结构化输出
def batch_structured_output():
    """批量结构化输出示例"""
    print("\n===== 示例5: 批量结构化输出 =====")
    
    # 准备批处理输入
    queries = [
        "请提供北京今天的天气信息",
        "请提供上海今天的天气信息"
    ]
    
    # 处理每个查询
    for i, query in enumerate(queries):
        print(f"查询 {i+1}:")
        response = get_structured_output(query, WeatherInfo)
        if response:
            print(f"  位置: {response.location}")
            print(f"  温度: {response.temperature}")
            print(f"  天气状况: {response.condition}")
        else:
            print("  未能获取有效的结构化输出")


# 示例6: 嵌套结构化输出
def nested_structured_output():
    """嵌套结构化输出示例"""
    print("\n===== 示例6: 嵌套结构化输出 =====")
    
    # 定义嵌套的Pydantic模型
    class DailyForecast(BaseModel):
        """每日天气预报"""
        date: str
        high_temp: str
        low_temp: str
        condition: str
    
    class WeeklyWeather(BaseModel):
        """周天气预报"""
        location: str
        forecasts: List[DailyForecast]
    
    # 使用自定义函数获取结构化输出
    prompt = "请提供北京未来三天的天气预报，每天包括日期、最高温度、最低温度和天气状况"
    response = get_structured_output(prompt, WeeklyWeather)
    
    if response:
        print(f"位置: {response.location}")
        print("未来三天天气预报:")
        for forecast in response.forecasts:
            print(f"  {forecast.date}: {forecast.low_temp}~{forecast.high_temp}, {forecast.condition}")
    else:
        print("未能获取有效的嵌套结构化输出")


# 示例7: 带验证的结构化输出
def validated_structured_output():
    """带验证的结构化输出示例"""
    print("\n===== 示例7: 带验证的结构化输出 =====")
    
    # 定义带验证的Pydantic模型
    class TemperatureRange(BaseModel):
        """温度范围模型"""
        location: str
        min_temp: float
        max_temp: float
        unit: str = "摄氏度"
        
        # 自定义验证
        def model_post_init(self, __context):
            if self.min_temp > self.max_temp:
                raise ValueError("最低温度不能高于最高温度")
    
    # 使用自定义函数获取结构化输出
    response = get_structured_output("请提供北京今天的温度范围", TemperatureRange)
    
    if response:
        print(f"位置: {response.location}")
        print(f"温度范围: {response.min_temp}~{response.max_temp}{response.unit}")
    else:
        print("未能获取有效的带验证结构化输出")


# 示例8: 流式结构化输出模拟
def stream_structured_output():
    """流式结构化输出模拟示例"""
    print("\n===== 示例8: 流式结构化输出模拟 =====")
    
    try:
        print("获取天气信息中...")
        response = get_structured_output("请提供北京今天的天气信息", WeatherInfo)
        
        # 模拟流式处理
        if response:
            print("\n流式处理结果:")
            print(f"位置: {response.location}")
            print(f"温度: {response.temperature}")
            print(f"天气状况: {response.condition}")
            print(f"建议: {response.suggestion}")
        else:
            print("未能获取有效的流式结构化输出")
    except Exception as e:
        print(f"流式结构化输出时出错: {e}")


# 示例9: 结合提示词的结构化输出
def prompt_structured_output():
    """结合提示词的结构化输出示例"""
    print("\n===== 示例9: 结合提示词的结构化输出 =====")
    
    # 创建一个包含详细指令的提示词
    detailed_prompt = """
    请分析以下用户问题并提供结构化信息:
    1. 提取用户提到的地点
    2. 提供该地点的天气信息
    3. 确保所有字段都有有效值
    
    用户问题: 我明天要去广州出差，需要了解天气情况
    """
    
    # 使用自定义函数获取结构化输出
    response = get_structured_output(detailed_prompt, WeatherInfo)
    
    if response:
        print("结合提示词的结构化输出结果:")
        print(f"  位置: {response.location}")
        print(f"  温度: {response.temperature}")
        print(f"  天气状况: {response.condition}")
        print(f"  建议: {response.suggestion}")
    else:
        print("未能获取有效的结合提示词结构化输出")


# 示例10: 动态选择结构化输出类型
def dynamic_structured_output():
    """动态选择结构化输出类型示例"""
    print("\n===== 示例10: 动态选择结构化输出类型 =====")
    
    # 创建一个函数来动态选择输出类型
    def dynamic_output_type(question):
        # 简单的意图识别
        if any(keyword in question for keyword in ["天气", "温度", "下雨", "晴天"]):
            return WeatherInfo
        elif any(keyword in question for keyword in ["产品", "价格", "商品"]):
            return ProductInfo
        elif any(keyword in question for keyword in ["书", "小说", "作者"]):
            return BookInfo
        else:
            return None
    
    # 创建处理函数
    def process_query(question):
        output_model = dynamic_output_type(question)
        
        if output_model:
            return get_structured_output(question, output_model)
        else:
            return f"无法确定输出类型，问题: {question}"
    
    # 测试不同类型的问题
    queries = [
        "请提供成都今天的天气信息",
        "请介绍一款热门的笔记本电脑",
        "请推荐一本编程相关的经典书籍"
    ]
    
    for query in queries:
        print(f"\n处理查询: {query}")
        result = process_query(query)
        if isinstance(result, str):
            print(result)
        elif result:
            print(f"结构化结果: {result}")
        else:
            print("未能获取有效的结构化输出")


if __name__ == "__main__":
    print("===== LangChain 结构化输出示例集 =====")
    
    # 运行所有示例
    basic_structured_output()
    structured_output_with_bind_tools()
    multi_type_structured_output()
    structured_output_chain()
    batch_structured_output()
    nested_structured_output()
    validated_structured_output()
    stream_structured_output()
    prompt_structured_output()
    dynamic_structured_output()
    
    print("\n所有结构化输出示例运行完毕！")