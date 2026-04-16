import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-eda26f20c01f42df8aadb6ea0d997f04"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 意图识别 -》 路由
# 用户提问 -》 类型1  类型2  类型3

sentiment_classification_agent = Agent(
    name="sentiment_classification_agent",
    model="qwen-max",
    instructions="""你是情感分析大师。回答任务前，请先说一句“我是情感分析大师，已准备好为您诊断情感：”。

你的核心任务是对提供的文本进行情感极性分类。
请将文本分类为以下三类之一：
1. "Positive"（正面）：表达赞美、高兴、满意、喜爱等积极情绪。
2. "Negative"（负面）：表达抱怨、愤怒、失望、悲伤等消极情绪。
3. "Neutral"（中性）：客观陈述事实，不带有明显的个人情感倾向。

要求：
1. 根据文本的整体基调进行判断。
2. 严格以 JSON 格式输出结果，不要包含其他无关内容。

期望的 JSON 输出格式示例：
{
  "sentiment": "Positive",
  "reason": "用户表达了对服务的满意和赞美"
}"""
)

entity_identification_agent = Agent(
    name="entity_identification_agent",
    model="qwen-max",
    instructions="""你是实体识别大师。回答任务前，请先说一句“我是实体识别大师，已准备好为您提取信息：”。
你的核心任务是从用户提供的文本中准确提取出特定的实体。
你需要识别的实体类型只有以下几种：
1. 人名 (Person)：文本中出现的人物姓名。
2. 地名 (Location)：国家、城市、详细地址等。
3. 机构名 (Organization)：公司、学校、政府机构等。
要求：
1. 如果某类实体没有在文本中出现，请返回空列表 []。
2. 请严格以 JSON 格式输出结果，不要包含其他解释性文字，以便系统解析。
期望的 JSON 输出格式示例：
{
  "Person": ["张三", "李四"],
  "Location": ["北京", "中关村"],
  "Organization": []
}"""
)

smart_agent = Agent(
    name="smart_agent",
    model="qwen-max",
    instructions="你是智能助手，对于非情感分类、实体识别的问题，由你来回答，回答问题的时候先告诉我你是谁。",
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="作为一个路由节点，你的任务是根据用户的输入意图，将任务转交给合适的 Agent。如果用户要求做情感分类或情感分析，请转交给 sentiment_classification_agent；如果用户要求做实体识别，请转交给 entity_identification_agent；如果是其他综合问题，请转交给 smart_agent。不要自己回答问题，只做路由(Handoff)操作。",
    handoffs=[sentiment_classification_agent, entity_identification_agent, smart_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    try:
        draw_graph(triage_agent, filename="路由Handoffs")
    except:
        print("绘制agent失败，默认跳过。。。")
    
    msg = input("你好，我是一个智能助手，请问有什么问题呢？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        # 强制在每一轮新的对话中，把入口重置为 triage_agent 进行重新路由分发
        agent = triage_agent


if __name__ == "__main__":
    asyncio.run(main())