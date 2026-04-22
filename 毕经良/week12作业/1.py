import sqlite3
import os
import re
import json

# 配置 DashScope (qwen-max) 相关环境变量
os.environ["OPENAI_API_KEY"] = "sk-eda26f20c01f42df8aadb6ea0d997f04" # 如果需要运行，请替换为您自己的 DashScope API Key
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import openai

# 数据库路径
DB_PATH = '04_SQL-Code-Agent-Demo/chinook.db'

def get_db_schema(db_path):
    """获取数据库中所有表的建表语句作为 Schema 上下文"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    schemas = cursor.fetchall()
    conn.close()
    
    schema_str = ""
    for s in schemas:
        if s[0]:
            schema_str += s[0] + "\n\n"
    return schema_str

def execute_sql(db_path, sql):
    """执行 SQL 并返回结果"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return True, result
    except Exception as e:
        conn.close()
        return False, str(e)

def ask_llm(messages):
    """调用 qwen-max 大模型"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model='qwen-max',
        messages=messages,
        temperature=0.1
    )
    return response.choices[0].message.content

def nl2sql_agent(question):
    """NL2SQL Agent 核心逻辑"""
    print(f"\n==========\n[问题] {question}")
    
    # 1. 获取数据库 schema
    schema = get_db_schema(DB_PATH)
    
    # 2. 构建 Prompt 让大模型生成 SQL
    prompt = f"""你是一个专业的数据库专家。请根据以下 SQLite 数据库的表结构，将用户的自然语言问题转换为 SQL 查询语句。
请直接输出 SQL 语句，不要包含任何额外的解释，也不要使用 markdown 代码块包裹（不要输出 ```sql ... ```）。

【数据库表结构】
{schema}

【用户问题】
{question}
"""
    
    # 获取 SQL
    sql = ask_llm([{"role": "user", "content": prompt}])
    sql = sql.strip().strip('`').replace('sql\n', '').strip()
    print(f"[生成的 SQL] {sql}")
    
    # 3. 执行 SQL
    success, result = execute_sql(DB_PATH, sql)
    if not success:
        print(f"[SQL 执行报错] {result}")
        return
        
    print(f"[SQL 执行结果] {result}")
    
    # 4. 根据结果生成自然语言回答
    answer_prompt = f"""你是一个智能问答助手。请根据用户的提问、查询使用的 SQL 以及查询结果，用自然语言给出一个清晰准确的回答。

【用户问题】
{question}

【执行的 SQL】
{sql}

【查询结果】
{result}

请直接给出自然语言回答：
"""
    answer = ask_llm([{"role": "user", "content": answer_prompt}])
    print(f"[最终回答] {answer}")


if __name__ == "__main__":
    # 需要回答的三个问题
    questions = [
        "数据库中总共有多少张表；",
        "员工表中有多少条记录；",
        "在数据库中所有客户个数和员工个数分别是多少？"
    ]
    
    for q in questions:
        nl2sql_agent(q)
