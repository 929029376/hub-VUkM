import traceback
from datetime import datetime

import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession, RunConfig, ModelSettings
from openai.types.responses import ResponseTextDeltaEvent, ResponseCreatedEvent, ResponseOutputItemDoneEvent, \
    ResponseFunctionToolCall
from agents.mcp import MCPServer, ToolFilterStatic, ToolFilterCallable
from agents import set_default_openai_api, set_tracing_disabled

# OpenAI-agent settings
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")

# SQLite session for conversation
session = SQLiteSession("conversation_123")

# Sidebar
with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = "sk-b872dc69e65543658466dc1bcfec3233"

    key = st.text_input('输入Token:', type='password', value=key)
    st.session_state['API_TOKEN'] = key

    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")


# Initial chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]
    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)


# ----------------------------
#   Streaming + RawEvent Logic
# ----------------------------
async def get_model_response1(prompt, model_name, use_tool):
    """
    :param prompt: 用户提问
    :param model_name:
    :param use_tool:
    :return:
    """
    async with MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        client_session_timeout_seconds=20,
    ) as mcp_server:

        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        result = Runner.run_streamed(agent, input=prompt, session=session)

        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                print(datetime.now(), "222", event)
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print(datetime.now(), "333", event)
                yield "raw", event.item.raw_item["output"]

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta



# 静态的tool筛选
async def get_model_response2(prompt, model_name, use_tool):
    news_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=["get_today_daily_news", "get_github_hot_news" ])
    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=["get_city_weather", "sentiment_classification"])

    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=news_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    external_client = AsyncOpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async with mcp_server1, mcp_server2:
        if use_tool:
            news_agent = Agent(
                name="News Assistant",
                instructions="Solve task, like 查询新闻",
                mcp_servers=[mcp_server2],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            tool_agnet = Agent(
                name="Tool Assistant",
                instructions="Solve task, like 查询天气",
                mcp_servers=[mcp_server1],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            agent = Agent(
                name="triage_agent",
                instructions="Handoff to the appropriate agent based on the language of the request.",
                handoffs=[news_agent, tool_agnet],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )


        result = Runner.run_streamed(agent, input=prompt, session=session, run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)))

        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                print(datetime.now(), "222", event)
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print(datetime.now(), "333", event)
                yield "raw", event.item.raw_item["output"]

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta

def mcp_news_callable_filter(context, tool) -> bool:
    return tool.name == "get_today_daily_news" or tool.name == "get_github_hot_news"

def mcp_tool_callable_filter(context, tool):
    return tool.name == "get_city_weather" or tool.name == "sentiment_classification"

def mcp_finance_callable_filter(context, tool):
    return tool.name in ["get_stock_price", "get_exchange_rate"]   # 金融工具
def mcp_bmi_callable_filter(context, tool):
    return tool.name == "calculate_bmi"             #BMI计算工具
def mcp_translate_callable_filter(context, tool):
    return tool.name == "translate_text"            #翻译工具
# 动态的工具的选择
async def get_model_response3(prompt, model_name, use_tool):
    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_tool_callable_filter,
        client_session_timeout_seconds=20,
    )

    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_news_callable_filter,
        client_session_timeout_seconds=20,
    )
    mcp_server_finance = MCPServerSse(
        name="SSE Python Server Finance",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_finance_callable_filter,  # 新增金融工具
        client_session_timeout_seconds=20,
    )
    mcp_server_bmi = MCPServerSse(
        name="SSE Python Server BMI",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_bmi_callable_filter, client_session_timeout_seconds=20,
    )
    mcp_server_translate = MCPServerSse(
        name="SSE Python Server Translate",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_translate_callable_filter, client_session_timeout_seconds=20,
    )
    external_client = AsyncOpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    async with mcp_server1, mcp_server2,mcp_server_finance,mcp_server_bmi,mcp_server_translate:

        if use_tool:
            news_agent = Agent(
                name="News Assistant",
                instructions="Solve task, like 查询新闻",
                mcp_servers=[mcp_server2],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            tool_agnet = Agent(
                name="Tool Assistant",
                instructions="Solve task, like 查询天气",
                mcp_servers=[mcp_server1],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            finance_agent = Agent(
                name="Finance Assistant",
                instructions="Solve tasks like querying stock prices, exchange rates, etc.",
                mcp_servers=[mcp_server_finance],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            bmi_agent = Agent(
                name="BMI Assistant",
                instructions="你是一个 BMI 计算助手。根据用户提供的身高（米）和体重（千克），调用 BMI 计算工具，返回计算出的 BMI 数值，并给出健康范围评估（如偏瘦、正常、超重、肥胖）。",
                mcp_servers=[mcp_server_bmi],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            translate_agent=Agent(
                name ="Translate Assistant",
                instructions="你是一个多语言翻译助手。根据用户输入的文本和目标语言，调用翻译工具进行准确翻译，并返回翻译结果。支持常见语言之间的互译。",
                mcp_servers=[mcp_server_translate],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
            agent = Agent(
                name="triage_agent",
                # instructions="Handoff to the appropriate agent based on the language of the request.",
                instructions="你是路由助手。根据用户请求的意图，将任务转交给对应的专业智能体：新闻(news_agent)、通用工具(tool_agent)、金融(finance_agent)、BMI计算(bmi_agent)、翻译(translate_agent)。",
                handoffs=[news_agent, tool_agnet,finance_agent,bmi_agent,translate_agent],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )


        result = Runner.run_streamed(agent, input=prompt, session=session, run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)))

        async for event in result.stream_events():
            print(datetime.now(), "111", event)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                print(datetime.now(), "222", event)
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    yield "argument", event.data.item

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print(datetime.now(), "333", event)
                yield "raw", event.item.raw_item["output"]

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta


# ----------------------------
#    Chat Interaction
# ----------------------------
if len(key) > 1:
    if prompt := st.chat_input():
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant streaming reply
        with st.chat_message("assistant"):
            placeholder = st.empty()

            with st.spinner("请求中..."):
                try:
                    # 把 streaming consumer 放成一个独立的 async 函数（使用局部 accumulated_text）
                    async def stream_output():
                        accumulated_text = ""
                        response_generator = get_model_response3(prompt, model_name, use_tool)

                        async for event_type, chunk in response_generator:
                            # Raw event（原始 delta），把它格式化为 code block，方便查看
                            if event_type == "argument":
                                # chunk 可能是 dict/list/其他对象 —— 转成字符串以防报错
                                formatted_raw = f"\n\n```json\n[RawArg]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            elif event_type == "raw":
                                # chunk 可能是 dict/list/其他对象 —— 转成字符串以防报错
                                formatted_raw = f"\n\n```json\n[RawEvent]\n{str(chunk)}\n```\n"
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            # 模型输出文本
                            elif event_type == "content":
                                # chunk 应该是 str（文本片段）
                                accumulated_text += chunk
                                placeholder.markdown(accumulated_text + "▌")

                        return accumulated_text

                    # 在同步上下文中运行 async generator
                    final_text = asyncio.run(stream_output())
                    # 最终渲染（去掉游标）
                    placeholder.markdown(final_text)

                except Exception as e:
                    error_msg = f"发生错误: {e}"
                    placeholder.error(error_msg)
                    final_text = error_msg
                    traceback.print_exc()

            # Save assistant reply to session state
            st.session_state.messages.append({"role": "assistant", "content": final_text})
