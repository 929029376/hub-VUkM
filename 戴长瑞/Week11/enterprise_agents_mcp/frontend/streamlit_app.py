import streamlit as st
import asyncio

from my_agents.triage_agent import get_triage_agent
from agents import Runner
from utils.validators import validate_input

st.set_page_config(page_title="企业级多Agent助手", layout="centered")
st.title("🧠 智能文本分析助手")
st.markdown("你可以输入一段文字，我会自动判断你需要**情感分析**还是**实体识别**。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


def is_valid_input(text: str) -> bool:
    result = validate_input(text)
    return result["valid"]


# 异步流式处理，实时显示过程
async def process_with_stream(user_input, status_container):
    """status_container 是一个可写入消息的容器（例如 st.empty()）"""
    status_container.info("🚀 初始化 Agent 系统...")
    triage_agent, mcp_servers = await get_triage_agent()

    status_container.info("🔌 连接 MCP 服务器...")
    async with mcp_servers[0], mcp_servers[1]:
        status_container.info("🧠 正在调用主 Agent 进行路由分析...")
        result = Runner.run_streamed(triage_agent, user_input)

        final_text = ""
        # 用于显示当前思考内容（模型增量输出）
        thinking_placeholder = st.empty()

        async for event in result.stream_events():
            # 处理模型文本输出（思考过程）
            if event.type == "raw_response_event" and hasattr(event, 'data'):
                if hasattr(event.data, 'delta'):
                    final_text += event.data.delta
                    # 实时显示模型的思考片段（限制长度）
                    thinking_placeholder.code(f"💭 {final_text[-200:]}", language="text")

            # 处理工具调用
            elif event.type == "run_item_stream_event" and event.name == "tool_call":
                tool_name = event.item.raw_item.get("name", "unknown")
                status_container.info(f"🔨 调用工具: {tool_name}")

            # 处理工具输出
            elif event.type == "run_item_stream_event" and event.name == "tool_output":
                output = str(event.item.raw_item["output"])[:200]
                status_container.info(f"📤 工具返回: {output}")

        thinking_placeholder.empty()  # 清除思考过程区域
        return final_text


user_input = st.chat_input("请输入一段文本...")
if user_input:
    if not is_valid_input(user_input):
        st.error("❌ 输入无效：不能为空，长度不能超过500字符，且不能包含非法字符")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            # 创建两个占位符：一个用于状态信息，一个用于思考过程
            status_placeholder = st.empty()

            try:
                response = asyncio.run(process_with_stream(user_input, status_placeholder))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                status_placeholder.error(f"发生错误: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"错误: {e}"})