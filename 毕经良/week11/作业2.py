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

    # TODO 基于用户名,在数据库中查询,返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


@mcp.tool
def calculate_leave_days(
    employee_id: Annotated[str, "Employee ID or name"],
    leave_type: Annotated[str, "Type of leave: 'annual' (年假), 'sick' (病假), 'personal' (事假)"]
):
    """Calculates remaining leave days for an employee based on their ID and leave type."""
    # Mock data - in production, this would query HR database
    leave_data = {
        "annual": {"total": 15, "used": 7, "remaining": 8},
        "sick": {"total": 10, "used": 2, "remaining": 8},
        "personal": {"total": 5, "used": 1, "remaining": 4}
    }
    
    if leave_type not in leave_data:
        return f"Invalid leave type. Available types: annual, sick, personal"
    
    result = leave_data[leave_type]
    return {
        "employee_id": employee_id,
        "leave_type": leave_type,
        "total_days": result["total"],
        "used_days": result["used"],
        "remaining_days": result["remaining"]
    }


@mcp.tool
def get_company_holidays(year: Annotated[int, "Year to query holidays (e.g., 2026)"]):
    """Retrieves official company holidays and public holidays for a specified year."""
    # Mock holiday data - in production, this would query company calendar system
    holidays = {
        2026: [
            {"date": "2026-01-01", "name": "元旦", "type": "public"},
            {"date": "2026-02-17", "name": "春节", "type": "public", "duration": "7 days"},
            {"date": "2026-04-05", "name": "清明节", "type": "public"},
            {"date": "2026-05-01", "name": "劳动节", "type": "public", "duration": "5 days"},
            {"date": "2026-06-25", "name": "端午节", "type": "public"},
            {"date": "2026-09-25", "name": "中秋节", "type": "public"},
            {"date": "2026-10-01", "name": "国庆节", "type": "public", "duration": "7 days"},
            {"date": "2026-12-25", "name": "公司年会", "type": "company"}
        ]
    }
    
    if year not in holidays:
        return f"Holiday data not available for year {year}. Available years: {list(holidays.keys())}"
    
    return {
        "year": year,
        "total_holidays": len(holidays[year]),
        "holidays": holidays[year]
    }


@mcp.tool
def search_internal_document(
    keyword: Annotated[str, "Search keyword or phrase"],
    doc_type: Annotated[str, "Document type: 'policy' (制度), 'technical' (技术文档), 'manual' (操作手册), 'all' (全部)"] = "all"
):
    """Searches internal company documents and knowledge base by keyword and document type."""
    # Mock document database - in production, this would query Elasticsearch or similar
    documents_db = [
        {"title": "员工考勤管理制度", "type": "policy", "tags": ["考勤", "制度", "人事"], "url": "/docs/policy/attendance.pdf"},
        {"title": "财务报销流程规范", "type": "policy", "tags": ["财务", "报销", "流程"], "url": "/docs/policy/reimbursement.pdf"},
        {"title": "API开发技术规范", "type": "technical", "tags": ["API", "开发", "规范"], "url": "/docs/tech/api-spec.md"},
        {"title": "数据库设计最佳实践", "type": "technical", "tags": ["数据库", "设计", "最佳实践"], "url": "/docs/tech/db-design.md"},
        {"title": "OA系统使用手册", "type": "manual", "tags": ["OA", "系统", "使用指南"], "url": "/docs/manual/oa-guide.pdf"},
        {"title": "会议室预订操作指南", "type": "manual", "tags": ["会议室", "预订", "行政"], "url": "/docs/manual/meeting-room.pdf"},
        {"title": "信息安全管理制度", "type": "policy", "tags": ["安全", "信息", "制度"], "url": "/docs/policy/security.pdf"},
        {"title": "代码审查规范", "type": "technical", "tags": ["代码", "审查", "规范"], "url": "/docs/tech/code-review.md"}
    ]
    
    # Filter by document type
    if doc_type != "all":
        filtered_docs = [doc for doc in documents_db if doc["type"] == doc_type]
    else:
        filtered_docs = documents_db
    
    # Search by keyword in title and tags
    results = []
    for doc in filtered_docs:
        if keyword.lower() in doc["title"].lower() or any(keyword.lower() in tag.lower() for tag in doc["tags"]):
            results.append(doc)
    
    if not results:
        return {
            "keyword": keyword,
            "doc_type": doc_type,
            "message": f"No documents found matching '{keyword}'",
            "total_results": 0,
            "results": []
        }
    
    return {
        "keyword": keyword,
        "doc_type": doc_type,
        "total_results": len(results),
        "results": results
    }
