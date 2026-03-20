from pydantic import BaseModel, Field
from typing import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044xxxxx0974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationRequest(BaseModel):
    """翻译任务请求解析"""
    source_language: Literal["英语", "中文", "日语", "韩语", "法语", "德语", "其他"] = Field(description="待翻译文本的原始语种")
    target_language: Literal["英语", "中文", "日语", "韩语", "法语", "德语", "其他"] = Field(description="目标翻译语种")
    text: str = Field(description="需要翻译的文本内容")

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

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """执行实际的翻译功能"""
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": f"请把以下{source_lang}文本翻译成{target_lang}：{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"翻译失败: {str(e)}"

print("=" * 50)
print("翻译智能体测试")
print("=" * 50)

print("\n【测试1】帮我把 good! 翻译成中文")
result = ExtractionAgent(model_name="qwen-plus").call("帮我把 good! 翻译成中文", TranslationRequest)
print(f"识别结果: 原始语种={result.source_language}, 目标语种={result.target_language}, 待翻译文本={result.text}")
if result:
    translation = translate_text(result.text, result.source_language, result.target_language)
    print(f"翻译结果: {translation}")

print("\n【测试2】请把 Hello, how are you? 翻译为日语")
result = ExtractionAgent(model_name="qwen-plus").call("请把 Hello, how are you? 翻译为日语", TranslationRequest)
print(f"识别结果: 原始语种={result.source_language}, 目标语种={result.target_language}, 待翻译文本={result.text}")
if result:
    translation = translate_text(result.text, result.source_language, result.target_language)
    print(f"翻译结果: {translation}")

print("\n【测试3】Help me translate 你好世界 to English")
result = ExtractionAgent(model_name="qwen-plus").call("Help me translate 你好世界 to English", TranslationRequest)
print(f"识别结果: 原始语种={result.source_language}, 目标语种={result.target_language}, 待翻译文本={result.text}")
if result:
    translation = translate_text(result.text, result.source_language, result.target_language)
    print(f"翻译结果: {translation}")
