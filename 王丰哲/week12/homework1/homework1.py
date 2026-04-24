from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NL2SQLResult:
    question: str
    sql: str
    rows: list[tuple[Any, ...]]
    answer: str


class ChinookNL2SQLAgent:
    """A small NL2SQL agent for the Chinook SQLite database.

    The flow follows the sql-agent notebook:
    1. inspect the database schema;
    2. translate a natural language question into SQL;
    3. check and execute SQL;
    4. rewrite the SQL result as a natural language answer.
    """

    TABLE_ALIASES = {
        "客户": "customers",
        "customer": "customers",
        "customers": "customers",
        "员工": "employees",
        "employee": "employees",
        "employees": "employees",
        "专辑": "albums",
        "album": "albums",
        "albums": "albums",
        "艺术家": "artists",
        "歌手": "artists",
        "artist": "artists",
        "artists": "artists",
        "歌曲": "tracks",
        "曲目": "tracks",
        "track": "tracks",
        "tracks": "tracks",
        "发票": "invoices",
        "invoice": "invoices",
        "invoices": "invoices",
        "发票明细": "invoice_items",
        "invoice_items": "invoice_items",
        "类型": "genres",
        "genre": "genres",
        "genres": "genres",
        "媒体类型": "media_types",
        "media_types": "media_types",
        "播放列表": "playlists",
        "playlist": "playlists",
        "playlists": "playlists",
    }

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path(__file__).with_name("chinook.db")
        if not self.db_path.exists():
            raise FileNotFoundError(f"Cannot find database file: {self.db_path}")

        self.conn = sqlite3.connect(self.db_path)
        self.tables = self._load_user_tables()

    def _load_user_tables(self) -> list[str]:
        sql = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
        return [row[0] for row in self.conn.execute(sql).fetchall()]

    def action(self, question: str) -> str:
        """Return only the final natural language answer."""
        return self.run(question).answer

    def run(self, question: str) -> NL2SQLResult:
        sql = self.nl2sql(question)
        self._check_sql(sql)
        rows = self.conn.execute(sql).fetchall()
        answer = self._rewrite_answer(question, sql, rows)
        return NL2SQLResult(question=question, sql=sql, rows=rows, answer=answer)

    def nl2sql(self, question: str) -> str:
        normalized = self._normalize(question)

        if self._mentions(normalized, "客户") and self._mentions(normalized, "员工"):
            return """
            SELECT
                (SELECT COUNT(*) FROM customers) AS customer_count,
                (SELECT COUNT(*) FROM employees) AS employee_count;
            """.strip()

        if self._asks_table_count(normalized):
            return """
            SELECT COUNT(*) AS table_count
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%';
            """.strip()

        table_name = self._extract_table_name(normalized)
        if table_name and self._asks_record_count(normalized):
            return f"SELECT COUNT(*) AS record_count FROM {table_name};"

        supported = "、".join(
            [
                "数据库中总共有多少张表",
                "员工表中有多少条记录",
                "在数据库中所有客户个数和员工个数分别是多少",
            ]
        )
        raise ValueError(f"暂不支持该问题。可尝试：{supported}")

    def _normalize(self, question: str) -> str:
        return re.sub(r"\s+", "", question.strip().lower())

    def _asks_table_count(self, question: str) -> bool:
        has_table_word = "表" in question or "table" in question
        has_count_word = any(word in question for word in ["多少", "几", "数量", "个数", "count"])
        asks_database_scope = any(word in question for word in ["数据库", "db", "database", "总共", "共有"])
        return has_table_word and has_count_word and asks_database_scope

    def _asks_record_count(self, question: str) -> bool:
        return any(word in question for word in ["多少条", "多少个", "几条", "记录", "数量", "个数", "count"])

    def _mentions(self, question: str, keyword: str) -> bool:
        return keyword in question or self.TABLE_ALIASES.get(keyword, "") in question

    def _extract_table_name(self, question: str) -> str | None:
        for alias, table_name in sorted(self.TABLE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in question and table_name in self.tables:
                return table_name
        return None

    def _check_sql(self, sql: str) -> None:
        compact_sql = sql.strip().rstrip(";").lower()
        if not compact_sql.startswith("select"):
            raise ValueError("Only SELECT SQL is allowed.")

        forbidden_words = ["insert", "update", "delete", "drop", "alter", "create", "replace", "pragma", "attach"]
        if any(re.search(rf"\b{word}\b", compact_sql) for word in forbidden_words):
            raise ValueError("Unsafe SQL was blocked.")

    def _rewrite_answer(self, question: str, sql: str, rows: list[tuple[Any, ...]]) -> str:
        if not rows:
            return "没有查询到结果。"

        normalized = self._normalize(question)
        first_row = rows[0]

        if self._mentions(normalized, "客户") and self._mentions(normalized, "员工"):
            return f"数据库中客户共有 {first_row[0]} 个，员工共有 {first_row[1]} 个。"

        if self._asks_table_count(normalized):
            return f"数据库中共有 {first_row[0]} 张业务表。"

        table_name = self._extract_table_name(normalized)
        if table_name == "employees":
            return f"员工表中共有 {first_row[0]} 条记录。"
        if table_name == "customers":
            return f"客户表中共有 {first_row[0]} 条记录。"
        if table_name:
            return f"{table_name} 表中共有 {first_row[0]} 条记录。"

        return f"查询结果为：{rows}"

    def close(self) -> None:
        self.conn.close()


def main() -> None:
    agent = ChinookNL2SQLAgent()
    questions = sys.argv[1:] or [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少",
    ]

    try:
        for question in questions:
            result = agent.run(question)
            print(f"问题：{result.question}")
            print(f"SQL：{result.sql}")
            print(f"回答：{result.answer}")
            print()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
