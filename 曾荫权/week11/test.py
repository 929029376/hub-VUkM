import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-8b47344f618342eaa3fdbab260e9e7a1"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from typing import Optional
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class Classification(BaseModel):
    category: str


# 守卫检查代理 - 》 本质也是通过大模型调用完成的
guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="请严格分析用户的问题属于哪一类：1. 如果是关于心情、情绪、心理状态的问题，输出 category='情感'。2. 如果是关于人名、地名、物品、概念定义的问题，输出 category='实体'。3. 如果都不是，输出 category='其他'。 json 返回",
    output_type=Classification,  # openai 官方推荐的一个语法，推荐大模型输出的格式类型，国内模型支持的不太好；
)

# 数学导师代理
emotions_agent = Agent(
    name="emotions Tutor",
    model="qwen-max",
    handoff_description="负责处理所有情感问题的心理医生代理。",
    instructions="您是专业的心理医生。请清晰地解释每一份情感的意义以及怎么应对当时的情感。",
)

# 历史导师代理
entitys_agent = Agent(
    name="Entity_Expert",
    model="qwen-max",
    handoff_description="负责处理所有实体、事实、定义类问题的专家。",
    instructions="你是一位严谨的【实体与事实分析专家】。你的任务是： 1. 识别用户输入中的具体实体（如人名、地名、物品、专业术语、历史事件等）。2. 对这些实体进行准确的定义和解释。3. 如果用户的问题不包含具体实体，或者你无法确认事实，请诚实地回答“根据现有知识，无法确认该信息”。【重要规则】- 严禁编造不存在的事实。- 严禁提及数学、计算或与实体无关的内容。- 请严格按照 JSON 格式输出结果。",
)


async def category_guardrail(ctx, agent, input_data):
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(Classification)

    # 关键替换 不替换会出现报错
    allowed_categories = ["情感", "实体"]
    tripwire_triggered = final_output.category not in allowed_categories

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )


# 先进行输入的校验 guardrail_agent
# triage_agent 判断 history_tutor_agent / math_tutor_agent
# history_tutor_agent 调用
triage_agent = Agent(
    name="Triage Agent",
    model="qwen-max",
    instructions="您的任务是根据用户的请求内容，判断应该将请求分派给 'Emotions Agent' 还是 'Entitys Agent'。",
    handoffs=[emotions_agent, entitys_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=category_guardrail),
    ],
)


async def main():
    print("--- 启动中文代理系统示例 ---")

    print("\n" + "=" * 50)
    print("=" * 50)
    """try:
        query = "我今天的心情很不好？"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)  # 异步运行  guardrail agent -》 triage agent -》 math agent
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**未识别到需要回答的问题", e)
"""
    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "稻城，长白山非常适合旅游"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**未识别到需要回答的问题", e)

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "你觉得明天深圳的天气怎么样？"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)  # 这行应该不会被执行
    except InputGuardrailTripwireTriggered as e:
        print("\n**未识别到需要回答的问题")
        print(e)


if __name__ == "__main__":
    asyncio.run(main())

    # try:
    #     draw_graph(triage_agent, filename="03_基础使用")
    # except:
    #     print("绘制agent失败，默认跳过。。。")

# python3 03_基础使用案例.py
