from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fcabuabuabuabub5a60974549b3", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

class TranslateAgent(BaseModel):
    """自动识别翻译任务：原始语种、目标语种、待翻译文本"""
    source_lang: str = Field(description="待翻译文本的原始语种，例如：中文、英文、日文、韩文、法文")
    target_lang: str = Field(description="需要翻译成的目标语种，例如：中文、英文、日文、韩文、法文")
    text: str = Field(description="需要翻译的原始文本内容")



if __name__ == '__main__':
    # 示例 1：标准翻译
    result1 = ExtractionAgent("qwen-plus").call(
        "帮我将good！翻译为中文",
        TranslateAgent
    )
    print("【翻译抽取结果1】")
    print(result1)

    print("-"*50)

    # 示例 2：中文翻英文
    result2 = ExtractionAgent("qwen-plus").call(
        "把我今天很开心翻译成英语",
        TranslateAgent
    )
    print("【翻译抽取结果2】")
    print(result2)

    print("-"*50)

    # 示例 3：日文翻译
    result3 = ExtractionAgent("qwen-plus").call(
        "おはよう翻译成中文是什么意思？",
        TranslateAgent
    )
    print("【翻译抽取结果3】")
    print(result3)
