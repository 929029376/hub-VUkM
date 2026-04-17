import re
from fastmcp import FastMCP

mcp = FastMCP(name="NER-Server")

@mcp.tool
def extract_entities(text: str) -> dict:
    """
    识别文本中的实体，返回格式: {"PERSON": [], "LOC": [], "ORG": []}
    """
    persons = re.findall(r'[李王张刘陈杨赵周吴郑][\u4e00-\u9fa5]{1,2}', text)
    locs = re.findall(r'[北京上海广州深圳杭州南京武汉成都重庆][市区县]?', text)
    orgs = re.findall(r'[中国美国日本]?[科技教育金融医疗能源][集团公司银行大学研究院]', text)

    return {
        "PERSON": list(set(persons)),
        "LOC": list(set(locs)),
        "ORG": list(set(orgs))
    }

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8902)