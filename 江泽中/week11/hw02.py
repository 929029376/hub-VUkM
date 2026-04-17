import re
import math
import random
import datetime
from typing import Annotated, Union
import requests

TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

# ---------- 原有工具 ----------
@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."],
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."],
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']
    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'
    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)
    count_positive = len(positive_matches)
    count_negative = len(negative_matches)
    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"

@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""
    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000

# ---------- 新增的 3 个自定义工具 ----------
@mcp.tool
def calculate(expression: Annotated[str, "Mathematical expression, e.g., '2 + 3 * 4'"]) -> str:
    """Evaluates a mathematical expression safely."""
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return f"Error: {name} is not allowed."
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@mcp.tool
def random_joke(category: Annotated[str, "Joke category: 'any', 'programming', 'dad'"] = "any") -> str:
    """Returns a random joke from a small local collection."""
    jokes = {
        "programming": [
            "为什么程序员总是混淆万圣节和圣诞节？因为 Oct 31 = Dec 25。",
            "一个 SQL 语句走进一家酒吧，看到两张桌子，问：我能 JOIN 你们吗？",
            "程序员的对象是：未定义。"
        ],
        "dad": [
            "为什么自行车不能站起来？因为它太累了。",
            "面包发烧了，结果变成了吐司。",
            "为什么数学书总是很忧伤？因为它的难题太多了。"
        ],
        "any": [
            "为什么手机也要戴眼镜？因为它总是掉像素。",
            "鱼为什么不敢打网球？因为会被网抓住。",
            "铅笔姓什么？姓‘削’，因为削铅笔。"
        ]
    }
    cat = category if category in jokes else "any"
    return random.choice(jokes[cat])

@mcp.tool
def current_time(timezone: Annotated[str, "Timezone name (e.g., 'Asia/Shanghai', 'America/New_York')"] = "Asia/Shanghai") -> str:
    """Returns the current date and time in the specified timezone."""
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ImportError:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC (pytz not installed)")
    except Exception as e:
        return f"时区错误：{str(e)}"

if __name__ == "__main__":
    mcp.run()