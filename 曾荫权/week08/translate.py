from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal
import openai
import json

client = openai.OpenAI(
    api_key="sk-8b47344abuabuabu260e9e7a1",
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
                    "description": response_model.model_json_schema().get('description', '提取结构化信息并执行任务'),
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema().get('required', []),
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

# 定义提取原始文本信息的模型
class SourceTextInfo(BaseModel):
    """提取原始语种和待翻译文本"""
    source_language: str = Field(description="识别出的原始语种，例如：中文、英文、日文等")
    content: str = Field(description="待翻译的原始文本内容")

# 定义最终翻译请求的模型
class Translation(BaseModel):
    """文本翻译智能体输出模型，包含翻译后的内容"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    original_content: str = Field(description="待翻译的原始文本内容")
    translated_content: str = Field(description="翻译后的文本内容")

def run_translation_agent():
    agent = ExtractionAgent(model_name="qwen-plus")
    
    print("=== 文本翻译智能体启动 ===")
    user_input = input("请输入您想要翻译的内容（例如：'把 Hello world 翻译成中文'，或者直接输入一段话）：\n")
    
    # 识别原始语种和文本内容
    print("\n正在识别原始语种...")
    source_info = agent.call(f"请识别这段话的原始语种和待翻译的文本内容：'{user_input}'", SourceTextInfo)
    
    if not source_info:
        print("抱歉，无法识别您的输入。")
        return

    print(f"识别结果 - 原始语种: {source_info.source_language}")
    print(f"待翻译文本: {source_info.content}")
    
    # 输出选项让用户选择目标语种
    print("\n请选择您想翻译成的目标语种：")
    options = ["中文", "英文", "日文", "德文"]
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    
    choice_idx = input("请输入选项数字（或直接输入语种名称）：")

    # 加一个兜底避免卡死
    target_lang = ""
    if choice_idx.isdigit() and 1 <= int(choice_idx) <= len(options):
        target_lang = options[int(choice_idx) - 1]
    else:
        target_lang = choice_idx
        
    if not target_lang:
        print("未选择有效的语种，默认翻译为 '中文'")
        target_lang = "中文"

    # 构造最终的 Translation 对象并执行翻译
    print(f"\n正在将文本从 {source_info.source_language} 翻译为 {target_lang}...")
    final_result = agent.call(
        f"请将以下文本从 {source_info.source_language} 翻译成 {target_lang}，并输出结构化数据。\n"
        f"原始文本：{source_info.content}", 
        Translation
    )
    
    if final_result:
        print("\n--- 翻译智能体输出结果 ---")
        print(f"原始语种: {final_result.source_language}")
        print(f"目标语种: {final_result.target_language}")
        print(f"原始内容: {final_result.original_content}")
        print(f"翻译内容: {final_result.translated_content}")
        print("-" * 25)
    else:
        print("构建翻译结果失败。")

if __name__ == "__main__":
    run_translation_agent()
