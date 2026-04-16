import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

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
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
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

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


# ===== 以下为自定义新增的 3 个工具 =====

@mcp.tool
def calculate_bmi(
    height_cm: Annotated[float, "身高，单位厘米(cm)，例如 170"],
    weight_kg: Annotated[float, "体重，单位千克(kg)，例如 65"]
):
    """根据身高(cm)和体重(kg)计算BMI指数，并给出健康评估。"""
    if height_cm <= 0 or weight_kg <= 0:
        return "输入无效，身高和体重必须为正数"
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi < 18.5:
        level = "偏瘦"
        advice = "建议适当增加营养摄入，加强力量训练。"
    elif bmi < 24:
        level = "正常"
        advice = "体重正常，请继续保持健康的生活方式。"
    elif bmi < 28:
        level = "偏胖"
        advice = "建议控制饮食，增加有氧运动。"
    else:
        level = "肥胖"
        advice = "建议尽快咨询医生，制定科学减重计划。"
    return {"BMI": bmi, "等级": level, "建议": advice}


@mcp.tool
def password_strength_check(password: Annotated[str, "要检测强度的密码字符串"]):
    """检测密码的安全强度等级，给出评分和改进建议。"""
    score = 0
    suggestions = []
    if len(password) >= 8:
        score += 1
    else:
        suggestions.append("长度至少8位")
    if len(password) >= 12:
        score += 1
    if re.search(r'[a-z]', password):
        score += 1
    else:
        suggestions.append("添加小写字母")
    if re.search(r'[A-Z]', password):
        score += 1
    else:
        suggestions.append("添加大写字母")
    if re.search(r'\d', password):
        score += 1
    else:
        suggestions.append("添加数字")
    if re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password):
        score += 1
    else:
        suggestions.append("添加特殊字符(!@#$%等)")

    levels = {0: "极弱", 1: "极弱", 2: "弱", 3: "一般", 4: "中等", 5: "强", 6: "极强"}
    return {"密码强度": levels.get(score, "极强"), "评分": f"{score}/6", "改进建议": suggestions if suggestions else ["密码强度很好，无需改进"]}


@mcp.tool
def text_statistics(text: Annotated[str, "要进行统计分析的文本内容"]):
    """对文本进行统计分析，返回字符数、中文字数、英文单词数、句子数、段落数等信息。"""
    total_chars = len(text)
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    digits = len(re.findall(r'\d', text))
    sentences = len(re.findall(r'[。！？.!?]+', text)) or (1 if text.strip() else 0)
    paragraphs = len([p for p in text.split('\n') if p.strip()])
    return {
        "总字符数": total_chars,
        "中文字数": chinese_chars,
        "英文单词数": english_words,
        "数字个数": digits,
        "句子数": sentences,
        "段落数": paragraphs,
    }
