from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai

client = openai.OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class Translation(BaseModel):
    """识别翻译的文本：原语言，目标语言，待翻译的文本"""
    source_language: Literal['中文','英语','德语','日语','法语'] = Field(description="源语言")
    target_language: Literal['中文','英语','德语','日语','法语'] = Field(description="目标语言")
    text: str = Field(description="待翻译的文本")

class ExtractionAgent:
    def __init__(self,model_name:str):
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
                "function":{
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters":{
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    }
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
            # print(response.choices[0].message)
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

translation_examples = [
    "把你好翻译成日语",
    "把谢谢翻译成法语",
    "把今天天气很好翻译成德语",
    "把I love you翻译成中文",
    "把Je t'aime翻译成中文",
    "把时间过得真快翻译成英语",
    "",
    "把请问地铁站在哪里翻译成日语",
]

for example in translation_examples:
    res = ExtractionAgent(model_name="qwen-plus").call(example, Translation)
    print(res)


# 运行结果
# source_language='中文' target_language='日语' text='你好'
# source_language='中文' target_language='法语' text='谢谢'
# source_language='中文' target_language='德语' text='今天天气很好'
# source_language='英语' target_language='中文' text='I love you'
# source_language='法语' target_language='中文' text="Je t'aime"
# source_language='中文' target_language='英语' text='时间过得真快'
# ERROR ChatCompletionMessage(content='Please provide the text you would like me to translate, along with the source and target languages.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None)
# None
# source_language='中文' target_language='日语' text='请问地铁站在哪里'
