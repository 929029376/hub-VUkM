

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerSse
from app_config.settings import settings

def create_sentiment_agent():
    mcp_server = MCPServerSse(
        name="Sentiment-MCP",
        params={"url": settings.mcp_sentiment_url},
        cache_tools_list=False,
    )
    external_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    agent = Agent(
        name="SentimentAgent",
        instructions="你是一个情感分析专家。用户会给你一段文本，你需要调用 classify_sentiment 工具返回情感结果（积极/消极/中性）。",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model=settings.model_name,
            openai_client=external_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    return agent, mcp_server