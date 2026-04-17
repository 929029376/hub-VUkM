import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-3d13848166aa4a5c902ad99e6c141e73"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner, trace
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


agent_1 = Agent(
    name="agent_1",
    model="qwen-max",
    instructions="对文本进行情感分类",
)

agent_2 = Agent(
    name = "agent_2",
    model = "qwen-max",
    instructions="对文本进行实体识别",
)

agent_main = Agent(
    name="main_agent",
    model="qwen-max",
    instructions="接收用户请求，选择哪一个Agent处理，并返回结果",
    handoffs=[agent_1, agent_2],
)

# 测试用例应该改为:
result = Runner.run_sync(agent_main, "这部电影太精彩了!")  # 测试情感分类
print("电影：",result.final_output)
# 或
result = Runner.run_sync(agent_main, "马云在杭州创立了阿里巴巴")  # 测试实体识别
print("马云：",result.final_output)

