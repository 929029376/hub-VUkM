import os
import openai
from pydantic import BaseModel, Field
from typing import List, Optional

# ================== 配置 OpenAI 客户端 ==================
api_key = os.getenv("DASHSCOPE_API_KEY", "sk-b872dc69e65543658466dc1bcfec3233")  # API Key 或设置环境变量
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# ================== 定义通用的 ExtractionAgent ==================
class ExtractionAgent:
    """利用函数调用从自然语言中提取结构化信息"""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model: BaseModel):
        """
        传入用户输入和 Pydantic 模型，返回模型实例
        """
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        # 动态构建 tools 参数
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get("title", response_model.__name__),
                    "description": schema.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", []),
                    },
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            # 提取工具调用参数
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                arguments = tool_calls[0].function.arguments
                return response_model.model_validate_json(arguments)
            else:
                print("模型未生成工具调用，原始返回：", response.choices[0].message.content)
                return None
        except Exception as e:
            print(f"调用失败：{e}")
            return None

# ================== 定义翻译任务专属模型 ==================
class TranslationRequest(BaseModel):
    """从翻译请求中提取源语言、目标语言和待翻译文本"""
    source_language: str = Field(
        description="原始语言，如'英语'、'中文'、'法语'等。如果用户未明确指定，则根据待翻译文本内容自动推断"
    )
    target_language: str = Field(
        description="目标语言，如'中文'、'英语'、'日语'等。由用户明确指定或从上下文推断"
    )
    text: str = Field(
        description="需要翻译的文本内容"
    )

# ================== 测试翻译智能体 ==================
if __name__ == "__main__":
    # 初始化智能体（使用支持函数调用的模型，如 qwen-plus）
    agent = ExtractionAgent(model_name="qwen-plus")

    # 测试用例列表
    test_cases = [
        "将good！翻译为中文",
        "把hello world从英语翻译成法语",
        "翻译bonjour为中文",
        "请帮我将'How are you?'译成德语",
        "把I love you 转成日语",
        "这个句子'今天天气真好'用英文怎么说？",
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {query}")
        result = agent.call(query, TranslationRequest)
        if result:
            print(f"  源语言: {result.source_language}")
            print(f"  目标语言: {result.target_language}")
            print(f"  待翻译文本: {result.text}")
        else:
            print("  提取失败")