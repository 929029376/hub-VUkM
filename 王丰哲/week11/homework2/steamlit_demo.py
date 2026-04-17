import traceback
from datetime import datetime
import json
import os

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

PROVIDER_CONFIGS = {
    "DashScope(阿里云兼容接口)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "models": ["qwen-flash", "qwen-max"],
        "key_hint": "请输入阿里云百炼 / DashScope 的 API Key，不能填 OpenAI 的 key。",
    },
    "OpenAI官方": {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4.1-mini", "gpt-4.1"],
        "key_hint": "请输入 OpenAI 官方 API Key，不能填 DashScope 的 key。",
    },
}

ENTERPRISE_TOOL_INSTRUCTIONS = """
你是企业职能助手。
- 对于企业内部数据或流程类问题，优先调用工具，不要凭空编造。
- 当用户问到年假、调休、工资、发薪时间、会议室预订、新闻、天气、电话归属地等信息时，请优先选择最合适的工具。
- 调用工具后，用中文给出简洁结论。
"""

CUSTOM_TOOL_NAMES = {
    "query_employee_leave_balance",
    "query_employee_payroll",
    "book_meeting_room",
}


def build_external_client(provider_name: str, api_key: str) -> AsyncOpenAI:
    if not api_key:
        raise ValueError("API Key 不能为空，请先在左侧输入与你选择的服务商对应的 API Key。")

    config = PROVIDER_CONFIGS[provider_name]
    if config["base_url"]:
        return AsyncOpenAI(api_key=api_key, base_url=config["base_url"])
    return AsyncOpenAI(api_key=api_key)

# SQLite session for conversation
session = SQLiteSession("conversation_123")

# Sidebar
with st.sidebar:
    st.title('职能AI+智能问答')
    provider_name = st.selectbox("选择服务商", list(PROVIDER_CONFIGS.keys()))
    provider_config = PROVIDER_CONFIGS[provider_name]

    session_key_name = f"API_TOKEN::{provider_name}"
    env_default_key = os.getenv(provider_config["env_key"], "")
    current_value = st.session_state.get(session_key_name, env_default_key)
    key = st.text_input('输入API KEY:', type='password', value=current_value)
    st.session_state[session_key_name] = key

    if len(key) > 1:
        st.success('API Token已经配置', icon='✅')
    else:
        st.info(provider_config["key_hint"])

    model_name = st.selectbox("选择模型", provider_config["models"])
    use_tool = st.checkbox("使用工具")
    st.caption(f"当前服务商：{provider_name}")
    st.caption("自定义工具示例：查询年假、查询工资、预订会议室")
    st.caption("示例提问：张三还有多少年假？")
    st.caption("示例提问：查询李四 2026-03 的工资")
    st.caption("示例提问：帮王敏预订 2026-04-19 10:00-11:00 的 A101 会议室，用于周会")


# Initial chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]): # 对话角色
        st.write(message["content"]) # 对外输出


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
        external_client = build_external_client(provider_name, key)

        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions=ENTERPRISE_TOOL_INSTRUCTIONS,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions=ENTERPRISE_TOOL_INSTRUCTIONS,
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

            # 最终大模型的返回
            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(datetime.now(), "444", event)
                yield "content", event.data.delta



# 静态的tool筛选
# get_model_response2 在 get_model_response1 基础上加入了mcp tool的静态选择，还加入了两个agent路由选择逻辑
async def get_model_response2(prompt, model_name, use_tool):
    # 静态的mcp tool 筛选方法
    news_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=["get_today_daily_news", "get_github_hot_news" ])
    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=[
        "get_city_weather",
        "sentiment_classification",
        "query_employee_leave_balance",
        "query_employee_payroll",
        "book_meeting_room",
    ])

    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=news_mcp_tools_filter, # 限定调用 news_mcp_tools_filter 其中的工具
        client_session_timeout_seconds=20,
    )

    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    external_client = build_external_client(provider_name, key)
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
                instructions="Solve task, like 查询天气、查询年假、查询工资、预订会议室",
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
    return tool.name in {
        "get_city_weather",
        "sentiment_classification",
        "query_employee_leave_balance",
        "query_employee_payroll",
        "book_meeting_room",
    }

# 动态的工具的选择
# get_model_response3 在 get_model_response2 基础上加入工具动态筛选
async def get_model_response3(prompt, model_name, use_tool):
    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_tool_callable_filter, # 动态tool筛选
        client_session_timeout_seconds=20,
    )

    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=mcp_news_callable_filter,
        client_session_timeout_seconds=20,
    )

    external_client = build_external_client(provider_name, key)
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
                instructions="Solve task, like 查询天气、查询年假、查询工资、预订会议室",
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
                instructions=ENTERPRISE_TOOL_INSTRUCTIONS,
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


def format_json_block(title, payload):
    try:
        if isinstance(payload, str):
            parsed = json.loads(payload)
        else:
            parsed = payload
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pretty = str(payload)
    return f"\n\n```json\n[{title}]\n{pretty}\n```\n"


def format_tool_call_block(tool_call: ResponseFunctionToolCall):
    tool_title = "CustomToolCall" if tool_call.name in CUSTOM_TOOL_NAMES else "ToolCall"
    payload = {
        "tool_name": tool_call.name,
        "arguments": json.loads(tool_call.arguments) if tool_call.arguments else {},
    }
    return format_json_block(tool_title, payload)


def format_tool_output_block(raw_output):
    tool_title = "CustomToolOutput" if isinstance(raw_output, dict) and raw_output.get("employee_name") else "ToolOutput"
    return format_json_block(tool_title, raw_output)


# ----------------------------
#    Chat Interaction
# ----------------------------
if len(key) > 1:
    if prompt := st.chat_input(): # 得到用户输入，判断输入是否为空
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 用户的角色暂时用户的输入
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

                        # 生成结果的迭代器
                        response_generator = get_model_response1(prompt, model_name, use_tool)

                        async for event_type, chunk in response_generator:

                            # Raw event（原始 delta），把它格式化为 code block，方便查看
                            if event_type == "argument":
                                formatted_raw = format_tool_call_block(chunk)
                                accumulated_text += formatted_raw
                                placeholder.markdown(accumulated_text + "▌")

                            elif event_type == "raw":
                                formatted_raw = format_tool_output_block(chunk)
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
                    error_text = str(e)
                    if "invalid_api_key" in error_text or "Incorrect API key provided" in error_text:
                        error_msg = (
                            "发生错误: 当前 API Key 无效，或者它和左侧选择的服务商不匹配。"
                            f" 当前服务商是：{provider_name}。请检查后重试。\n\n原始错误: {error_text}"
                        )
                    else:
                        error_msg = f"发生错误: {error_text}"
                    placeholder.error(error_msg)
                    final_text = error_msg
                    traceback.print_exc()

            # Save assistant reply to session state
            st.session_state.messages.append({"role": "assistant", "content": final_text})
