"""
NL2SQL Agent —— 基于 Chinook SQLite 数据库的自然语言问答 Agent
参照 sql-agent.ipynb，使用通义千问 API + ReAct 框架实现。

支持三类提问：
  1. 数据库中总共有多少张表
  2. 员工表中有多少条记录
  3. 数据库中所有客户个数和员工个数分别是多少
"""

import os
import json
import sqlite3
import re
import textwrap
from typing import Union
from sqlalchemy import create_engine, inspect, text
from openai import OpenAI


DB_PATH = r"E:\BaiduNetdiskDownload\八斗学院\第12周-ChatBI数据智能问答\Week12\04_SQL-Code-Agent-Demo\chinook.db"
QWEN_API_KEY = input("请输入通义千问 API Key：")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus" 



class DBParser:
    """数据库解析工具，支持 SQLite / MySQL"""

    def __init__(self, db_url: str) -> None:
        self.engine = create_engine(db_url, echo=False)
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

    # ── 工具方法 ──────────────────────────────────

    def list_tables(self) -> list[str]:
        """返回数据库中所有用户表名列表"""
        return self.table_names

    def get_schema(self, table_name: str) -> str:
        """返回指定表的字段名和类型（格式化文本）"""
        columns = self.inspector.get_columns(table_name)
        lines = [f"  {col['name']} ({col['type']})" for col in columns]
        return f"Table: {table_name}\n" + "\n".join(lines)

    def get_all_schemas(self) -> str:
        """返回所有表的 schema 概要，作为 system prompt 上下文"""
        parts = []
        for t in self.table_names:
            parts.append(self.get_schema(t))
        return "\n\n".join(parts)

    def execute_sql(self, sql: str) -> Union[list, str]:
        """执行 SQL，返回结果列表；出错返回错误字符串"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                return [list(r) for r in rows]
        except Exception as e:
            return f"SQL执行错误: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "列出数据库中所有表的名称，返回表名列表",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "获取指定表的字段名和类型信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "表名，必须是数据库中存在的表"
                    }
                },
                "required": ["table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "执行一条 SQL 查询语句，返回查询结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的 SQL 查询语句（SELECT 语句）"
                    }
                },
                "required": ["sql"]
            }
        }
    }
]

class NL2SQLAgent:
    """
    基于 ReAct（Reason + Act）框架的 NL2SQL Agent。
    流程：用户提问 → LLM 推理 → 调用 Tool → 观察结果 → 循环直到给出最终答案
    """

    def __init__(self, db_parser: DBParser, api_key: str, base_url: str, model: str):
        self.parser = db_parser
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 系统提示：包含数据库 schema 信息
        all_schemas = self.parser.get_all_schemas()
        self.system_prompt = textwrap.dedent(f"""
            你是一个专业的数据库智能问答助手（NL2SQL Agent）。
            你可以通过调用工具查询 SQLite 数据库来回答用户的自然语言问题。

            可用工具：
            - list_tables：列出所有表
            - get_schema：获取某张表的字段结构
            - execute_sql：执行 SQL 查询语句

            数据库中所有表的结构如下：
            {all_schemas}

            回答规则：
            1. 先思考需要哪些信息，选择合适的工具获取
            2. 根据工具返回结果，生成准确的 SQL
            3. 执行 SQL 后，用自然语言回答用户
            4. 回答要简洁明确，包含具体数字
        """).strip()

    def _dispatch_tool(self, tool_name: str, tool_args: dict) -> str:
        """分发工具调用，返回字符串结果"""
        if tool_name == "list_tables":
            tables = self.parser.list_tables()
            return f"数据库中共有 {len(tables)} 张用户表，表名为：{tables}"
        elif tool_name == "get_schema":
            return self.parser.get_schema(tool_args.get("table_name", ""))
        elif tool_name == "execute_sql":
            sql = tool_args.get("sql", "")
            result = self.parser.execute_sql(sql)
            return f"SQL: {sql}\n结果: {result}"
        else:
            return f"未知工具: {tool_name}"

    def ask(self, question: str, max_rounds: int = 8) -> str:
        """
        对用户问题执行 ReAct 循环，返回最终自然语言答案。

        参数：
            question: 用户的自然语言提问
            max_rounds: 最大推理轮次，防止死循环
        """
        print(f"\n{'=' * 60}")
        print(f"🤔 用户提问：{question}")
        print(f"{'=' * 60}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        for round_num in range(max_rounds):
            print(f"\n--- 第 {round_num + 1} 轮推理 ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # ① 如果 LLM 直接给出了最终答案（没有 tool_calls）
            if not msg.tool_calls:
                final_answer = msg.content or "（无回答）"
                print(f"✅ 最终回答：{final_answer}")
                return final_answer

            # ② 处理 tool_calls
            # 先把 assistant 消息加入历史
            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"🔧 调用工具：{tool_name}  参数：{tool_args}")

                tool_result = self._dispatch_tool(tool_name, tool_args)

                print(f"📊 工具结果：{tool_result}")

                # 将工具结果追加到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        return "（已达最大推理轮次，未能给出答案）"

def main():
    print("正在连接数据库并初始化 Agent...")
    parser = DBParser(f"sqlite:///{DB_PATH}")
    agent = NL2SQLAgent(
        db_parser=parser,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        model=QWEN_MODEL
    )
    print(f"✅ 数据库已连接，共加载 {len(parser.table_names)} 张表")

    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？"
    ]

    results = []
    for i, q in enumerate(questions, 1):
        answer = agent.ask(q)
        results.append((i, q, answer))

    # ── 汇总输出 ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("📋 汇总结果")
    print(f"{'=' * 60}")
    for idx, question, answer in results:
        print(f"\n【提问{idx}】{question}")
        print(f"【回答{idx}】{answer}")

    print(f"\n{'=' * 60}")
    print("全部提问回答完毕！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
