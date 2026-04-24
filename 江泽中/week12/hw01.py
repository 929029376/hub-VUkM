import re
import traceback
from DBParser import DBParser
import jwt
import time
import requests

# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

def ask_glm(question, nretry=5):
    if nretry == 0:
        return None

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("83e5bc58555d8bac289e27bac50f8afc.Khk1JjCxb8MJN8Mi", 1000)
    }
    data = {
        "model": "glm-3-turbo",
        "p": 0.5,
        "messages": [{"role": "user", "content": question}]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()
    except:
        return ask_glm(question, nretry-1)


class NL2SQLAgent:
    def __init__(self, db_path: str = "chinook.db"):
        self.db_url = f"sqlite:///{db_path}"
        self.parser = DBParser(self.db_url)
        self.schema_text = self._build_schema_description()
        self.sql_gen_prompt = """你是一个 SQL 专家，根据用户的问题和数据库结构，生成一条正确的 SQLite 查询语句。

### 数据库结构：
{schema}

### 用户问题：
{question}

### 要求：
- 只输出 SQL 语句，不要包含任何解释、注释或 markdown 格式。
- SQL 语句以分号结尾。
- 如果问题涉及计数，请使用 COUNT(*) 并返回数字。
- 只生成 SELECT 查询，不要执行修改操作。
- 表名和字段名严格使用上面给出的名称。

SQL：
"""
        self.sql_reflection_prompt = """你生成的 SQL 语句执行时发生了错误。
错误信息：{error}
原 SQL：{sql}
数据库结构：
{schema}
用户问题：{question}
请修正 SQL，只输出修正后的 SQL 语句（不要输出其他内容）。
SQL：
"""
        self.answer_prompt = """用户的问题是：{question}
对应的 SQL 查询是：{sql}
查询结果是：{result}
请将结果用自然语言回答用户，直接给出答案。例如：“数据库中共有 8 张表。” 或 “员工表中共有 8 条记录。” 等。
回答："""
        self.max_retries = 3

    def _build_schema_description(self) -> str:
        lines = []
        for table in self.parser.table_names:
            lines.append(f"表名: {table}")
            fields_df = self.parser.get_table_fields(table)
            for idx, row in fields_df.iterrows():
                col_info = f"  - {row['name']} ({row['type']})"
                if row.get('nullable'):
                    col_info += " 可为空"
                if row.get('primary_key'):
                    col_info += " 主键"
                lines.append(col_info)
            lines.append("")
        relations = self.parser.get_data_relations()
        if not relations.empty:
            lines.append("外键关系：")
            for _, fk in relations.iterrows():
                lines.append(
                    f"  {fk['constrained_table']}.{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
        return "\n".join(lines)

    def _extract_sql(self, llm_output: str) -> str:
        cleaned = re.sub(r'```sql\s*|```\s*', '', llm_output, flags=re.IGNORECASE)
        match = re.search(r'(SELECT\s+.*?;?)', cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            if not sql.endswith(';'):
                sql += ';'
            return sql
        return cleaned.strip()

    def ask(self, question: str) -> str:
        prompt = self.sql_gen_prompt.format(schema=self.schema_text, question=question)
        llm_out = ask_glm(prompt)
        if not llm_out or 'choices' not in llm_out:
            return "模型调用失败，请重试。"
        sql = self._extract_sql(llm_out['choices'][0]['message']['content'])

        for attempt in range(self.max_retries):
            ok, err_msg = self.parser.check_sql(sql)
            if not ok:
                reflect_prompt = self.sql_reflection_prompt.format(
                    error=err_msg, sql=sql, schema=self.schema_text, question=question
                )
                llm_out = ask_glm(reflect_prompt)
                if llm_out and 'choices' in llm_out:
                    sql = self._extract_sql(llm_out['choices'][0]['message']['content'])
                continue

            try:
                result_raw = self.parser.execute_sql(sql)
                if not result_raw:
                    result_str = "查询结果为空。"
                elif len(result_raw) == 1 and len(result_raw[0]) == 1:
                    result_str = str(result_raw[0][0])
                else:
                    result_str = "\n".join([str(row) for row in result_raw[:10]])

                answer_prompt = self.answer_prompt.format(question=question, sql=sql, result=result_str)
                llm_answer = ask_glm(answer_prompt)
                if llm_answer and 'choices' in llm_answer:
                    return llm_answer['choices'][0]['message']['content']
                else:
                    return f"执行结果是：{result_str}"
            except Exception as e:
                reflect_prompt = self.sql_reflection_prompt.format(
                    error=traceback.format_exc(), sql=sql, schema=self.schema_text, question=question
                )
                llm_out = ask_glm(reflect_prompt)
                if llm_out and 'choices' in llm_out:
                    sql = self._extract_sql(llm_out['choices'][0]['message']['content'])
                continue

        return "抱歉，多次尝试后仍无法生成正确的 SQL 查询。"


if __name__ == "__main__":
    agent = NL2SQLAgent("chinook.db")
    questions = [
        "数据库中总共有多少张表；",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]
    for q in questions:
        print(f"\n问题: {q}")
        answer = agent.ask(q)
        print(f"回答: {answer}")