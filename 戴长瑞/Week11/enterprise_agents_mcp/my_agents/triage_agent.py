
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner
from .sentiment_agent import create_sentiment_agent
from .ner_agent import create_ner_agent
from app_config.settings import settings

async def get_triage_agent():
    sentiment_agent, sentiment_mcp = create_sentiment_agent()
    ner_agent, ner_mcp = create_ner_agent()

    external_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    triage_agent = Agent(
        name="TriageAgent",
        instructions="""
你是一个路由助手。根据用户输入的内容判断应该交给哪个子Agent处理：
- 如果用户要求分析情感（如“这句话是正面还是负面”、“帮我判断一下情绪”），交给 SentimentAgent。
- 如果用户要求提取实体（如“找出这段话里的人名、地名”），交给 NERAgent。
- 如果用户没有明确指定，但文本中包含明显的情感词汇或实体，也按上述规则分配。
- 其他情况，请回复“无法处理，请明确要求情感分析或实体识别”。
""",
        handoffs=[sentiment_agent, ner_agent],
        model=OpenAIChatCompletionsModel(
            model=settings.model_name,
            openai_client=external_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    return triage_agent, [sentiment_mcp, ner_mcp]