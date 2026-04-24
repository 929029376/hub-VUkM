# 第十二周作业2答案

## 1. 什么是前后端分离？

前后端分离是把系统拆成两个相对独立的部分：

- 前端：负责界面展示、交互、状态管理（例如聊天框、按钮、历史列表）。
- 后端：负责业务逻辑、模型调用、数据库读写、权限控制。
- 通信方式：前端通过 HTTP/SSE/WebSocket 请求后端 API，后端返回 JSON 或流式数据。

在 06-stock-bi-agent 中：

- Streamlit 页面属于前端（demo 目录）。
- FastAPI 服务属于后端（main_server.py + routers + services）。
- 前端通过 `/v1/chat/`、`/v1/chat/init`、`/v1/chat/get` 等接口和后端交互。

## 2. 历史对话如何存储，以及如何将历史对话作为大模型的下一次输入？

项目里是“双轨存储”：

### A. 业务数据库存储（用于查询和展示）

- 会话级信息：`ChatSessionTable`（会话 id、标题、开始时间等）。
- 消息级信息：`ChatMessageTable`（role、content、时间、反馈等）。
- 每轮对话时：
  - 用户消息先写入数据库。
  - 助手回复完成后再写入数据库。
- 前端重新进入会话时，会调用后端接口读取历史消息并回显。

### B. Agent 运行时记忆（用于下一轮上下文）

- 后端使用 `AdvancedSQLiteSession(session_id=..., db_path="./assert/conversations.db")`。
- 在调用 `Runner.run_streamed(..., session=session)` 时，把该 session 传给 Agent。
- 下一轮同一个 `session_id` 再请求时，历史上下文会自动带入模型输入。

简化理解：

- `sever.db` 里存“可管理、可展示”的业务聊天记录；
- `conversations.db` 里存 openai-agent 的会话记忆；
- 同一个 `session_id` 把两边串起来，实现“能看历史 + 模型也记得历史”。
