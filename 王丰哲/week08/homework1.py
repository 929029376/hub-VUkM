from pydantic import BaseModel, Field
import openai

client = openai.OpenAI(
    api_key="sk-831e7efabuabuabuabu8a4e9547f",
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
        except Exception:
            print("ERROR:", response.choices[0].message)
            return None


class TranslationRequest(BaseModel):
    """自动识别翻译任务中的原始语种、目标语种和待翻译文本"""
    source_language: str = Field(description="原始语种，例如英文、中文、日文")
    target_language: str = Field(description="目标语种，例如中文、英文、法文")
    text: str = Field(description="待翻译的原始文本内容")


result = ExtractionAgent(model_name="qwen-plus").call(
    "帮我将constrast, good, embed翻译为中文",
    TranslationRequest
)

print(result)
