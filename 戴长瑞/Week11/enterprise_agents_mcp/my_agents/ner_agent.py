
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerSse
from app_config.settings import settings

def create_ner_agent():
    mcp_server = MCPServerSse(
        name="NER-MCP",
        params={"url": settings.mcp_ner_url},
        cache_tools_list=False,
    )
    external_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    agent = Agent(
        name="NERAgent",
        instructions="你是一个命名实体识别专家。用户会给你一段文本，你需要调用 extract_entities 工具返回 JSON 格式的实体列表，包含 PERSON、LOC、ORG。",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model=settings.model_name,
            openai_client=external_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    return agent, mcp_server