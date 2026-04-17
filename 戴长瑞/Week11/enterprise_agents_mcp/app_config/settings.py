from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI 兼容 API 配置
    openai_api_key: str = "sk-b872dc69e65543658466dc1bcfec3233"
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-flash"

    # MCP 服务端点
    mcp_sentiment_url: str = "http://localhost:8901/sse"
    mcp_ner_url: str = "http://localhost:8902/sse"

    # 日志级别
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()