import streamlit as st
import sqlite3
import re
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

# ----------------------------- 配置 ----------------------------------
API_KEY = "6a24be37272f42b88fe8979f786ea15f.GONqjyq9U1QP3L4U"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DB_PATH = "chinook.db"

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ----------------------------- 数据库工具 -----------------------------
def get_db_connection():
    """创建新的数据库连接，允许跨线程使用（只读查询）"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(ttl=3600)
def get_schema() -> str:
    """提取数据库完整 schema（表名、列名、类型、主键、外键）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_lines = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_defs = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            is_nullable = "NULL" if col[3] else "NOT NULL"
            is_pk = "PRIMARY KEY" if col[5] else ""
            col_defs.append(f"    {col_name} {col_type} {is_nullable} {is_pk}".strip())

        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        fk_lines = []
        for fk in fks:
            fk_lines.append(f"    FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")

        schema_lines.append(f"Table: {table}")
        schema_lines.extend(col_defs)
        if fk_lines:
            schema_lines.extend(fk_lines)
        schema_lines.append("")

    conn.close()
    return "\n".join(schema_lines)


def execute_single_sql(sql: str, conn) -> Tuple[Optional[Tuple[List[Any], List[str]]], Optional[str]]:
    """执行单条 SELECT 语句，返回 (rows, col_names) 或错误信息"""
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description] if cursor.description else []
        return (rows, col_names), None
    except Exception as e:
        return None, f"SQL 执行错误: {str(e)}"


def execute_sql(sql: str, conn) -> Tuple[Optional[Tuple[List[Any], List[str]]], Optional[str]]:
    """
    安全执行 SQL 查询，支持多条 SELECT 语句（以分号分隔）。
    如果多条语句的结果都是单行单列，则合并为一行多列；
    否则报错要求使用单条语句。
    """
    sql = sql.strip()
    if not sql:
        return None, "SQL 为空。"

    # 检查是否以 SELECT 开头（忽略前导空白）
    if not sql.upper().startswith("SELECT"):
        return None, "只允许执行 SELECT 查询语句。"

    # 拆分多条语句（简单按分号分割，忽略注释和字符串内的分号，但足够应对 LLM 生成的简单情况）
    statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]

    if len(statements) == 1:
        return execute_single_sql(statements[0], conn)

    # 多条语句：分别执行，尝试合并结果
    results = []
    col_names = []
    for i, stmt in enumerate(statements):
        if not stmt.upper().startswith("SELECT"):
            return None, f"第 {i + 1} 条语句不是 SELECT 查询。"
        res, err = execute_single_sql(stmt, conn)
        if err:
            return None, f"执行第 {i + 1} 条语句失败: {err}"
        rows, cols = res
        if len(rows) != 1 or len(rows[0]) != 1:
            return None, f"多条语句仅支持每个子查询返回单个数值（单行单列），但第 {i + 1} 条返回了 {len(rows)} 行 {len(rows[0]) if rows else 0} 列。"
        results.append(rows[0][0])
        col_names.append(cols[0] if cols else f"col_{i + 1}")

    # 合并为一行多列
    merged_rows = [tuple(results)]
    return (merged_rows, col_names), None


# ----------------------------- LLM 调用 -----------------------------
def generate_sql(question: str, schema: str, prev_errors: list = None) -> str:
    """
    使用 LLM 将自然语言问题转换为 SQL 查询。
    如果提供了之前的错误，会进行自我修正。
    """
    system_prompt = """你是一个 SQL 专家。根据用户的问题和数据库 schema，生成正确的 SQLite 查询语句。

重要要求：
- 只返回 SQL 语句，不要有任何额外解释或注释。
- 仅使用 SELECT 查询。
- 必须生成 **单一的 SELECT 语句**，不要使用分号分隔多条语句。
- 如果需要同时获取多个聚合结果（例如客户数和员工数），请使用子查询：SELECT (SELECT COUNT(*) FROM table1) AS alias1, (SELECT COUNT(*) FROM table2) AS alias2;
- 确保列名和表名与 schema 完全一致（大小写敏感）。
- 如果问题中提到的概念在 schema 中不存在，请生成一条能返回提示信息的 SQL（如 SELECT '无法回答'）。
"""

    user_prompt = f"""数据库 Schema:
{schema}

用户问题: {question}

请生成单一的 SQLite 查询语句:"""

    if prev_errors:
        user_prompt += f"\n\n之前生成的 SQL 执行失败，错误信息: {prev_errors[-1]}\n请修正 SQL，确保生成单一 SELECT 语句（可使用子查询合并多个结果）。"

    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        sql = response.choices[0].message.content.strip()
        # 清理 markdown 代码块
        sql = re.sub(r"```sql\n?(.*?)\n?```", r"\1", sql, flags=re.DOTALL)
        sql = re.sub(r"```\n?(.*?)\n?```", r"\1", sql, flags=re.DOTALL)
        return sql
    except Exception as e:
        st.error(f"LLM 调用失败: {str(e)}")
        return ""


def answer_question(question: str, max_retries: int = 2) -> Tuple[str, str]:
    """
    完整流程：生成 SQL -> 执行 -> 返回自然语言答案 和 最后生成的SQL。
    返回值: (答案文本, 最后生成的SQL)
    """
    schema = get_schema()
    conn = get_db_connection()
    last_sql = ""
    try:
        errors = []
        for attempt in range(max_retries + 1):
            sql = generate_sql(question, schema, errors if errors else None)
            last_sql = sql
            if not sql:
                return "无法生成有效的 SQL 语句。", last_sql

            result, err = execute_sql(sql, conn)
            if err is None:
                rows, col_names = result
                if not rows:
                    return "查询结果为空。", last_sql
                if len(rows) == 1 and len(col_names) == 1:
                    return f"答案是：{rows[0][0]}", last_sql
                else:
                    return format_result_as_text(rows, col_names), last_sql
            else:
                errors.append(err)
                if attempt == max_retries:
                    return f"多次尝试后仍无法生成正确 SQL。最后错误: {err}\n最后生成的 SQL: {sql}", last_sql
        return "未知错误。", last_sql
    finally:
        conn.close()


def format_result_as_text(rows, col_names) -> str:
    """将查询结果格式化为自然语言（简单表格）"""
    if len(rows) == 0:
        return "未找到相关数据。"
    header = " | ".join(col_names)
    sep = "-+-".join(["---"] * len(col_names))
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="NL2SQL Agent - Chinook 数据库问答", layout="wide")
st.title("Chinook 数据库问答助手 (NL2SQL)")
st.markdown("输入自然语言问题，系统自动生成 SQL 并返回答案。")

# 示例问题
st.sidebar.header("示例问题")
example_questions = [
    "数据库中总共有多少张表；",
    "员工表中有多少条记录；",
    "在数据库中所有客户个数和员工个数分别是多少；",
    "列出所有来自美国的客户姓名和邮箱；",
    "哪个艺术家的作品最多？",
]
for q in example_questions:
    if st.sidebar.button(q, key=q):
        st.session_state.question = q

# 用户输入
user_question = st.text_area("请输入您的问题:", value=st.session_state.get("question", ""), height=100)
if st.button("提交问题", type="primary"):
    if user_question.strip():
        with st.spinner("正在理解问题并查询数据库..."):
            answer, last_sql = answer_question(user_question)
        st.success("查询完成")
        st.subheader("答案")
        st.text(answer)
        with st.expander("查看生成的 SQL（调试）"):
            if last_sql:
                st.code(last_sql, language="sql")
                print(f"[DEBUG] 生成的 SQL:\n{last_sql}")  # 控制台打印
            else:
                st.info("未生成有效的 SQL")
    else:
        st.warning("请输入问题。")

# 显示数据库 schema（折叠）
with st.expander("📖 当前数据库 Schema"):
    st.text(get_schema())

st.sidebar.markdown("---")
st.sidebar.info(
    "**使用说明**\n"
    "- 基于智谱 GLM-4 模型将自然语言转为 SQL\n"
    "- 仅支持 SELECT 查询，保证数据安全\n"
    "- 自动处理错误并尝试修正 SQL（最多重试 2 次）\n"
    "- 支持多条 SELECT 语句自动合并为一行多列（仅限单值结果）\n"
)