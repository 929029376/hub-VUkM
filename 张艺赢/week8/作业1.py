from pydantic import BaseModel, Field
from typing_extensions import Literal

import openai

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044xxxxxxx974549b3",
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


class TranslationRequest(BaseModel):
    """解析用户翻译请求，提取原始语种、目标语种和待翻译文本"""
    source_language: Literal["中文", "英文", "日文", "韩文", "法文", "德文", "西班牙文", "俄文", "自动识别"] = Field(
        description="原始语种，即待翻译文本的语言类型"
    )
    target_language: Literal["中文", "英文", "日文", "韩文", "法文", "德文", "西班牙文", "俄文"] = Field(
        description="目标语种，即用户希望翻译成的语言类型"
    )
    text_to_translate: str = Field(
        description="待翻译的文本内容"
    )


if __name__ == "__main__":
    agent = ExtractionAgent(model_name="qwen-plus")
    
    test_cases = [
        "帮我将good！翻译为中文",
        "请把'你好世界'翻译成英文",
        "How are you? 翻译成中文",
        "把这段话翻译成日语：今天天气真好",
        "I love programming, 请翻译",
    ]
    
    for text in test_cases:
        print(f"\n输入: {text}")
        result = agent.call(text, TranslationRequest)
        print(f"结果: {result}")
        print("-" * 50)
