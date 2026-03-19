from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本质是function call
# 可以传入多个待选函数，让大模型选择其中一个
# 传的是我们的函数的描述，让大模型选择，生成调用这个函数的传入参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "Ticket",
            "description": "根据用户提供的信息查询火车时刻",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "description": "要查询的火车日期",
                        "title": "Date",
                        "type": "string",
                    },
                    "departure": {
                        "description": "出发城市或车站",
                        "title": "Departure",
                        "type": "string",
                    },
                    "destination": {
                        "description": "要查询的火车日期",
                        "title": "Destination",
                        "type": "string",
                    },
                },
                "required": ["date", "departure", "destination"],
            },
        },
    }
]

"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现信息信息抽取
指定写的tool的格式
"""
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Translate(BaseModel):
    """帮我将good！翻译为中文 -》 原始语种、目标语种，待翻译的文本"""
    resourceLanguage: str = Field(description="原始语种")
    resourceTxt: str = Field(description="待翻译的文本")
    targetLanguage: str = Field(description="目标语种")
result = ExtractionAgent(model_name = "qwen-plus").call("帮我将good！翻译为中文", Translate)
print(result)


