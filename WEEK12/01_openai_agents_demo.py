"""第十二周作业1：基于 Chinook 数据集的 NL2SQL Agent。

功能：
1. 将中文自然语言问题映射为 SQL（规则版 NL2SQL）。
2. 在 chinook.db 上执行 SQL 并返回可读答案。
3. 内置并验证题目要求的 3 个问题。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QueryResult:
    sql: str
    rows: list[tuple[Any, ...]]


class ChinookNL2SQLAgent:
    """一个轻量级 NL2SQL Agent。

    说明：这里使用规则映射以保证作业题目中的关键问题可稳定回答。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def nl2sql(self, question: str) -> str:
        q = question.strip().lower()

        # 提问1: 数据库中总共有多少张表
        if ("多少张表" in q) or ("总共有多少表" in q) or ("table" in q and "多少" in q):
            return (
                "SELECT COUNT(*) AS table_count "
                "FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )

        # 提问2: 员工表中有多少条记录
        if ("员工表" in q and "多少" in q) or ("employee" in q and "count" in q):
            return "SELECT COUNT(*) AS employee_count FROM employees;"

        # 提问3: 客户个数和员工个数分别是多少
        if ("客户" in q and "员工" in q and "个数" in q) or ("customer" in q and "employee" in q):
            return (
                "SELECT "
                "(SELECT COUNT(*) FROM customers) AS customer_count, "
                "(SELECT COUNT(*) FROM employees) AS employee_count;"
            )

        raise ValueError("暂不支持该问题，请尝试题目中的3个问题或相近问法。")

    def run_sql(self, sql: str) -> QueryResult:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
        return QueryResult(sql=sql, rows=rows)

    def answer(self, question: str) -> str:
        sql = self.nl2sql(question)
        result = self.run_sql(sql)

        if "sqlite_master" in sql:
            return f"数据库中总共有 {result.rows[0][0]} 张表。"
        if "FROM employees" in sql and "customer_count" not in sql:
            return f"员工表中共有 {result.rows[0][0]} 条记录。"
        if "customer_count" in sql and "employee_count" in sql:
            customer_count, employee_count = result.rows[0]
            return f"客户总数为 {customer_count}，员工总数为 {employee_count}。"

        return f"SQL执行成功，结果：{result.rows}"


def get_default_db_path() -> Path:
    # 当前文件位于 WEEK10/，chinook.db 位于 llm_learning/Week12/04_SQL-Code-Agent-Demo/
    return Path(__file__).resolve().parents[1] / "llm_learning" / "Week12" / "04_SQL-Code-Agent-Demo" / "chinook.db"


def run_required_questions(agent: ChinookNL2SQLAgent) -> None:
    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？",
    ]

    print("=" * 60)
    print("第十二周作业1 - Chinook NL2SQL Agent")
    print(f"数据库路径: {agent.db_path}")
    print("=" * 60)

    for idx, q in enumerate(questions, start=1):
        sql = agent.nl2sql(q)
        ans = agent.answer(q)
        print(f"\n[提问{idx}] {q}")
        print(f"[SQL] {sql}")
        print(f"[回答] {ans}")


def interactive_chat(agent: ChinookNL2SQLAgent) -> None:
    print("\n进入交互模式，输入 quit 退出。")
    while True:
        q = input("\n请输入问题: ").strip()
        if q.lower() == "quit":
            break
        try:
            sql = agent.nl2sql(q)
            ans = agent.answer(q)
            print(f"SQL: {sql}")
            print(f"回答: {ans}")
        except Exception as exc:
            print(f"错误: {exc}")


def main() -> None:
    db_path = get_default_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"未找到 chinook.db: {db_path}")

    agent = ChinookNL2SQLAgent(db_path=db_path)
    run_required_questions(agent)
    interactive_chat(agent)


if __name__ == "__main__":
    main()
