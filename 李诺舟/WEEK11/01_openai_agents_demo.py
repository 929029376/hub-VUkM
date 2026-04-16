"""
openai-agents 多Agent路由示例
- 主Agent（triage）：接收用户输入，根据意图选择子Agent
- 子Agent 1：情感分类
- 子Agent 2：实体识别

安装：pip install openai-agents
要求：Python >= 3.10，需设置 OPENAI_API_KEY 环境变量
"""

import asyncio
from agents import Agent, Runner

# ── 子Agent 1：情感分类 ──
sentiment_agent = Agent(
    name="sentiment_agent",
    instructions=(
        "你是一个情感分类专家。"
        "对用户提供的文本进行情感分析，输出分类结果（正面/负面/中性）并给出简要理由。"
        "只做情感分类，不要做其他任务。"
    ),
    handoff_description="当用户需要对文本进行情感分析、情绪判断时，转交给此Agent",
)

# ── 子Agent 2：实体识别 ──
ner_agent = Agent(
    name="ner_agent",
    instructions=(
        "你是一个命名实体识别（NER）专家。"
        "对用户提供的文本进行实体识别，提取出人名、地名、组织名、时间、数值等实体，"
        "以列表形式输出每个实体及其类型。"
        "只做实体识别，不要做其他任务。"
    ),
    handoff_description="当用户需要从文本中提取人名、地名、组织名等命名实体时，转交给此Agent",
)

# ── 主Agent：路由分发 ──
triage_agent = Agent(
    name="triage_agent",
    instructions=(
        "你是一个任务分发Agent。根据用户的请求意图，将任务转交给合适的子Agent：\n"
        "1. 如果用户想对文本做情感分析/情绪判断，转交给 sentiment_agent\n"
        "2. 如果用户想从文本中提取实体（人名、地名、机构等），转交给 ner_agent\n"
        "不要自己回答问题，直接handoff给对应的Agent。"
    ),
    handoffs=[sentiment_agent, ner_agent],
)


async def main():
    print("=" * 50)
    print("多Agent路由系统（情感分类 / 实体识别）")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n请输入你的请求：")
        if user_input.strip().lower() == "quit":
            break

        result = await Runner.run(triage_agent, input=user_input)
        print(f"\n[最终处理Agent]: {result.last_agent.name}")
        print(f"[输出结果]:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
