import streamlit as st
import asyncio
import json
import traceback
from mcp import ClientSession
from mcp.client.sse import sse_client

st.set_page_config(page_title="MCP 工具调试器", layout="wide")
st.title("🔧 MCP 工具调试面板")

# ---------- 1. 连接配置 ----------
with st.sidebar:
    st.header("连接设置")
    server_url = st.text_input(
        "MCP SSE 服务器地址",
        value="http://localhost:8900/sse",
        help="你的 fastmcp 服务器 SSE 端点，例如 http://localhost:8900/sse"
    )
    refresh_btn = st.button("🔄 刷新工具列表")


# ---------- 2. 获取工具列表 ----------
@st.cache_resource(ttl=60, show_spinner=False)
def load_tools(url):
    async def _fetch():
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    try:
        return asyncio.run(_fetch())
    except Exception as e:
        st.error(f"❌ 连接服务器失败: {e}")
        return None


if refresh_btn:
    st.cache_resource.clear()
    tools = load_tools(server_url)
else:
    tools = load_tools(server_url)

if tools is None:
    st.stop()

st.success(f"✅ 成功连接，共发现 **{len(tools)}** 个工具")

# ---------- 3. 工具选择与参数表单 ----------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📦 工具列表")
    tool_names = [tool.name for tool in tools]
    selected_name = st.selectbox("选择要测试的工具", tool_names)
    selected_tool = next(t for t in tools if t.name == selected_name)

    with st.expander("📄 工具描述"):
        st.markdown(f"**{selected_tool.name}**")
        st.write(selected_tool.description or "无描述")
        st.json(selected_tool.inputSchema)

with col2:
    st.subheader(f"🧪 调用测试：`{selected_tool.name}`")

    # 根据 inputSchema 动态生成参数输入控件
    schema = selected_tool.inputSchema
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    args = {}
    for param, info in properties.items():
        param_type = info.get("type", "string")
        label = f"{param} ({param_type})"
        help_text = info.get("description", "")
        is_required = param in required
        if param_type == "number":
            val = st.number_input(label, value=0.0, step=0.1, key=param, help=help_text)
        elif param_type == "integer":
            val = st.number_input(label, value=0, step=1, key=param, help=help_text)
        elif param_type == "boolean":
            val = st.checkbox(label, key=param, help=help_text)
        else:
            val = st.text_input(label, value="", key=param, help=help_text)
        args[param] = val

    # 可选：显示最终将要发送的 JSON 参数
    st.caption(f"将要发送的参数：`{json.dumps(args, ensure_ascii=False)}`")

    if st.button("🚀 调用工具", type="primary"):
        # 检查必需参数
        missing = [p for p in required if p not in args or args[p] in (None, "", 0.0) and not isinstance(args[p], bool)]
        if missing:
            st.error(f"缺少必需参数: {', '.join(missing)}")
        else:
            with st.spinner("调用中，请稍候..."):
                try:
                    async def call():
                        async with sse_client(server_url) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                result = await session.call_tool(selected_tool.name, arguments=args)
                                return result


                    result = asyncio.run(call())

                    st.success("✅ 调用成功")

                    # 展示结果（原始内容）
                    st.subheader("📤 返回结果")
                    if hasattr(result, 'content'):
                        # MCP 标准返回格式
                        for item in result.content:
                            if item.type == "text":
                                st.text(item.text)
                            else:
                                st.json(item.model_dump())
                    else:
                        st.json(result)

                    # 额外：显示原始对象（调试用）
                    with st.expander("🔍 原始返回对象"):
                        st.write(result)

                except Exception as e:
                    st.error(f"❌ 调用失败: {e}")
                    with st.expander("📋 详细错误堆栈"):
                        st.code(traceback.format_exc())