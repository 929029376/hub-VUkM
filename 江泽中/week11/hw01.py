from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from typing import List
import asyncio


# ==============================================
# 1. 定义两个子Agent
# ==============================================

# --- 1.1 情感分类子Agent (Sentiment Agent)---
class SentimentResult(BaseModel):
    """情感分类的结构化输出模型"""
    sentiment: str  # 情感极性，可以是 "positive", "negative", "neutral"
    confidence: float  # 置信度，范围 0-1
    analysis: str  # 简要的分析原因


sentiment_agent = Agent(
    name="Sentiment Analyst",
    instructions="你是一个情感分析专家。请分析用户输入文本的情感，并以JSON格式返回结果。",
    output_type=SentimentResult,  # 使用Pydantic模型约束输出格式
    model="qwen-max"
)


# --- 1.2 实体识别子Agent (NER Agent)---
class Entity(BaseModel):
    """单个实体的模型"""
    text: str  # 实体文本
    type: str  # 实体类型，如 "PERSON", "ORG", "LOC", "DATE" 等
    confidence: float  # 置信度


class NERResult(BaseModel):
    """实体识别结果的结构化输出模型"""
    entities: List[Entity]


ner_agent = Agent(
    name="NER Expert",
    instructions="你是一个命名实体识别专家。请从文本中提取所有命名实体，并按类型分类。",
    output_type=NERResult,
    model="qwen-max"
)

# ==============================================
# 2. 定义主控Agent (Orchestrator Agent)
# ==============================================
orchestrator_agent = Agent(
    name="Orchestrator",
    instructions=(
        "你是一个任务协调员。你的职责是根据用户的请求，将其委派给最合适的专业智能体。"
        "请仔细判断用户请求的意图："
        "1. 如果用户询问或要求进行'情感分析'、'情绪'、'情感'等，请将任务移交给 'Sentiment Analyst'。"
        "2. 如果用户询问或要求进行'实体识别'、'NER'、'命名实体'、'提取人名、地名'等，请将任务移交给 'NER Expert'。"
    ),
    # 关键：handoffs列表告诉主控Agent它可以移交任务给哪些子Agent
    handoffs=[sentiment_agent, ner_agent],
)


# ==============================================
# 3. 运行并测试系统
# ==============================================

async def main():
    # 测试用例1：情感分析
    user_input_1 = "我今天真的非常开心，这个框架太棒了！"
    print(f"用户输入: {user_input_1}")
    result_1 = await Runner.run(orchestrator_agent, user_input_1)

    # 检查最终输出是否来自移交
    if result_1.final_output_as(SentimentResult):
        sentiment_result = result_1.final_output_as(SentimentResult)
        print(f"-> 情感分析结果: {sentiment_result}")
    else:
        print(f"-> 主控直接回答: {result_1.final_output}")

    print("-" * 50)

    # 测试用例2：实体识别
    user_input_2 = "请告诉我，苹果公司的CEO蒂姆·库克在纽约说了什么？"
    print(f"用户输入: {user_input_2}")
    result_2 = await Runner.run(orchestrator_agent, user_input_2)

    if result_2.final_output_as(NERResult):
        ner_result = result_2.final_output_as(NERResult)
        print(f"-> 实体识别结果: {ner_result}")
    else:
        print(f"-> 主控直接回答: {result_2.final_output}")


# 如果你是在同步环境下运行，可以使用 Runner.run_sync
# 例如：
# result = Runner.run_sync(orchestrator_agent, "分析文本的情感")
# print(result.final_output)

if __name__ == "__main__":


    asyncio.run(main())