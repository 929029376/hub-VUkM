from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

LOCAL_ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-max"


def load_local_env_file() -> None:
    if not LOCAL_ENV_FILE.exists():
        return

    for raw_line in LOCAL_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configure_runtime() -> None:
    # This example defaults to the OpenAI-compatible DashScope endpoint used in the course.
    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    os.environ.setdefault("OPENAI_MODEL", DEFAULT_MODEL)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)


load_local_env_file()
configure_runtime()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(
        description="Overall sentiment label."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1.",
    )
    rationale: str = Field(description="Short reason for the label.")
    keywords: List[str] = Field(
        default_factory=list,
        description="A few words that influenced the sentiment decision.",
    )


class EntityItem(BaseModel):
    text: str = Field(description="The entity surface form.")
    type: Literal["person", "organization", "location", "date", "product", "event", "other"] = Field(
        description="Entity type."
    )
    normalized: Optional[str] = Field(
        default=None,
        description="Canonical form if one is available.",
    )


class EntityResult(BaseModel):
    entities: List[EntityItem] = Field(
        default_factory=list,
        description="Detected named entities.",
    )
    summary: str = Field(description="Short summary of the extraction result.")


sentiment_agent = Agent(
    name="sentiment_agent",
    model=DEFAULT_MODEL,
    handoff_description="Analyze the sentiment of a text.",
    instructions="""你是一个情感分析助手。请分析用户文本的情感状态。
    请务必以 JSON 格式输出，并且必须包含以下三个字段：
    - "sentiment": 情感分析结果（例如 positive, negative, neutral）
    - "confidence": 你对这个判断的置信度，格式为0到1之间的数字（例如 0.95）
    - "rationale": 给出你这样判断的理由（例如：因为用户提到了"很仔细"等正面词汇）
    """,
    output_type=SentimentResult,
)

entity_agent = Agent(
    name="Entity Recognition Agent",
    model=DEFAULT_MODEL,
    handoff_description="Extract named entities from a text.",
    instructions=(
        "你只做实体识别。请抽取文本中的人名、组织、地点、时间、产品和事件，并返回结构化结果。"
        "不要做情感分类，也不要回答和实体识别无关的问题。"
        "请务必以 JSON 格式输出，包含以下字段：\n"
        '- "entities": 实体列表，每个实体包含 "text"（原文）、"type"（类型，取值为 person/organization/location/date/product/event/other）、"normalized"（标准化名称，可为 null）\n'
        '- "summary": 对识别结果的简短总结'
    ),
    output_type=EntityResult,
)

router_agent = Agent(
    name="Router Agent",
    model=DEFAULT_MODEL,
    instructions=(
        "你是一个路由 agent。你必须把每次请求只转给下面两个子 agent 中的一个，不要自己直接回答。"
        "如果用户要做情感分析、情绪分类、正负面判断、态度判断，就转给 Sentiment Agent。"
        "如果用户要做实体识别、命名实体识别、抽取人名、地点、组织、时间、产品，就转给 Entity Recognition Agent。"
        "如果请求有点模糊，根据用户最明确的意图选择一个最合适的子 agent。"
    ),
    handoffs=[sentiment_agent, entity_agent],
)


def format_output(output: object) -> str:
    if isinstance(output, BaseModel):
        return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)
    return str(output)


async def main() -> None:
    print("输入一条请求，主 agent 会自动路由到情感分类或实体识别。直接回车退出。")
    print("示例：")
    print("  - 请分析这段文本的情感：这个产品太好用了，我很满意。")
    print("  - 请提取这段文本中的实体：OpenAI 在旧金山发布了新模型。")

    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            break

        result = await Runner.run(router_agent, user_input)
        print(f"\n由 {result.last_agent.name} 返回：")
        print(format_output(result.final_output))


if __name__ == "__main__":
    asyncio.run(main())
