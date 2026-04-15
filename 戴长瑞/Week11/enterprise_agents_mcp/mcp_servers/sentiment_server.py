import re
from fastmcp import FastMCP

mcp = FastMCP(name="Sentiment-Analysis-Server")

@mcp.tool
def classify_sentiment(text: str) -> str:
    """
    对输入文本进行情感分类，返回: 积极 / 消极 / 中性
    """
    pos_keywords = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意', '高兴']
    neg_keywords = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍', '伤心']

    pos_count = sum(1 for kw in pos_keywords if kw in text)
    neg_count = sum(1 for kw in neg_keywords if kw in text)

    if pos_count > neg_count:
        return "积极"
    elif neg_count > pos_count:
        return "消极"
    else:
        return "中性"

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8901)