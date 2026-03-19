from pydantic import BaseModel, Field
from typing import Optional
import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class TranslationRequest(BaseModel):
    """文本翻译参数提取 3"""
    source_language: str = Field(
        description="原始文本的语言（自动识别），例如：English, Chinese, French"
    )
    target_language: str = Field(
        description="目标翻译语言，例如：Chinese, English, Japanese"
    )
    text: str = Field(
        description="需要翻译的原始文本内容"
    )
    confidence: float = Field(
        description="识别置信度（0.0-1.0）",
        ge=0.0,
        le=1.0
    )


class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def extract_translation_params(self, user_input: str) -> Optional[TranslationRequest]:
        tools = [{
            "type": "function",
            "function": {
                "name": TranslationRequest.__name__,
                "description": TranslationRequest.__doc__ or "",
                "parameters": TranslationRequest.model_json_schema()
            }
        }]

        messages = [
            {"role": "system",
             "content": "精准提取翻译需求：识别原文、原语言、目标语言。语言用标准英文名（如Chinese/English），置信度反映识别确定性。"},
            {"role": "user", "content": user_input}
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1  # 降低随机性，提高参数提取稳定性
            )

            tool_call = response.choices[0].message.tool_calls[0]
            return TranslationRequest.model_validate_json(tool_call.function.arguments)

        except Exception as e:
            print(f"参数提取失败: {str(e)}")
            return None

    def translate(self, user_input: str) -> str:
        params = self.extract_translation_params(user_input)
        if not params:
            return "无法识别翻译需求，请明确说明'将[文本]翻译为[语言]'"
        validation = []
        if "chinese" in params.target_language.lower() and any(c.isascii() for c in params.text):
            validation.append("✓ 原文含英文字符，目标语言为中文（合理）")
        elif "english" in params.target_language.lower() and not any(c.isascii() for c in params.text):
            validation.append("✓ 原文为中文，目标语言为英文（合理）")

        return (
            f"翻译参数提取成功！\n"
            f"{'─' * 40}\n"
            f"原文:      '{params.text}'\n"
            f"原语言:    {params.source_language}\n"
            f"目标语言:  {params.target_language}\n"
            f"置信度:    {params.confidence:.0%}\n"
            f"{'─' * 40}\n"
            f"验证提示:  {'; '.join(validation) if validation else '参数逻辑合理'}\n"
            f"示例请求:  translate(text='{params.text}', from='{params.source_language}', to='{params.target_language}')"
        )

if __name__ == "__main__":
    agent = TranslationAgent()

    test_cases = [
        "帮我将good！翻译为中文",  # 全角感叹号
        "把'你好世界'翻译成英文",
        "我需要把法语的'Bonjour'翻译成日语",
        "translate this sentence to Japanese: 今天天气真好",  # 混合语言
        "把'Hello, 世界!' 翻译成 Spanish"  # 多标点+大小写
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}")
        print(f"测试用例 {i}: {query}")
        print(f"{'=' * 50}")
        print(agent.translate(query))
