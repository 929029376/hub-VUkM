import os
import asyncio
from pydantic import BaseModel
from typing import Literal

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled

# ====================== 配置阿里云百炼 ======================
os.environ["OPENAI_API_KEY"] = input('请输入你的阿里云密钥:')
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ====================== 输出结构定义 ======================

class TaskType(BaseModel):
    """主Agent用于判断任务类型的结构"""
    task_type: Literal["sentiment", "entity", "unknown"]


class SentimentOutput(BaseModel):
    """情感分类输出结构"""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    explanation: str


class EntityOutput(BaseModel):
    """实体识别输出结构（简化版，避免 schema 报错）"""
    entities: str  # 用字符串描述实体，更稳定
    explanation: str


# ====================== 子 Agent 定义 ======================

# 子Agent 1: 情感分类
sentiment_agent = Agent(
    name="Sentiment Analysis Agent",
    model="qwen-max",
    handoff_description="负责对文本进行情感分类的专家代理。",
    instructions="""
    你是专业的情感分析专家。
    对用户提供的文本进行情感分类：positive（积极）、negative（消极）、neutral（中性）。
    请给出置信度（0.0-1.0）和简短解释。
    请以 JSON 格式输出，必须严格遵循 SentimentOutput 结构。
    """,
    output_type=SentimentOutput,
)

# 子Agent 2: 实体识别
entity_agent = Agent(
    name="Named Entity Recognition Agent",
    model="qwen-max",
    handoff_description="负责对文本进行实体识别的专家代理。",
    instructions="""
    你是专业的命名实体识别专家。
    请从文本中提取重要实体（人名、组织、地点、时间等），并说明类型。
    请以清晰的文字描述所有实体。
    请以 JSON 格式输出，必须严格遵循 EntityOutput 结构。
    """,
    output_type=EntityOutput,
)

# ====================== 守卫检查代理 ======================

guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="""
    判断用户的问题是否需要进行情感分类或实体识别。
    如果是情感分析相关（心情、情绪、观点），'task_type' 为 "sentiment"；
    如果是提取实体相关（人名、地名、公司、时间等），'task_type' 为 "entity"；
    其他情况 'task_type' 为 "unknown"。
    请以 JSON 格式返回。
    """,
    output_type=TaskType,
)


async def task_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为情感分类或实体识别任务。
    如果是 unknown，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(TaskType)

    tripwire_triggered = final_output.task_type == "unknown"

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )


# ====================== 主 Agent（Triage Agent） ======================

triage_agent = Agent(
    name="Triage Agent",
    model="qwen-max",
    instructions="""
    你的任务是根据用户输入的内容，判断应该将请求分派给 'Sentiment Analysis Agent' 还是 'Named Entity Recognition Agent'。
    - 如果用户想分析情感、情绪、观点倾向 → 分派给 Sentiment Analysis Agent
    - 如果用户想提取文本中的实体（人名、地名、机构、时间等） → 分派给 Named Entity Recognition Agent
    """,
    handoffs=[sentiment_agent, entity_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=task_guardrail),
    ],
)


# ====================== 主程序 ======================

async def main():
    print("--- 启动文本分析多代理系统 ---")

    print("\n" + "=" * 60)
    print("=" * 60)

    test_cases = [
        "我今天心情特别好，感觉一切都很顺利！",
        "苹果公司总部位于美国加州库比蒂诺。",
        "这部电影真是太烂了，浪费时间！",
        "明天深圳的天气怎么样？",
        "张伟昨天在上海和华为的李经理签了合同。",
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\n测试 {i}: **用户提问:** {query}")
        print("-" * 60)

        try:
            result = await Runner.run(triage_agent, query)
            print("\n**✅ 流程通过，最终输出:**")
            print(result.final_output)

        except InputGuardrailTripwireTriggered as e:
            print("\n**❌ 守卫阻断触发:** 输入被阻断，因为它不是情感分类或实体识别任务。")
            print(e)
        except Exception as e:
            print(f"\n**⚠️ 发生异常:** {e}")

        print("\n" + "=" * 60)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
