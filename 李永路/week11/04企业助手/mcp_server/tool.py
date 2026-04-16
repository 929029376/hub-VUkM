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


@mcp.tool
def get_employee_info(employee_name: Annotated[str, "员工姓名或工号"]):
    """Retrieves employee information including department, position, email, and phone number based on employee name or ID."""
    # 模拟员工数据库
    employees_db = {
        "张三": {"name": "张三", "department": "技术部", "position": "高级工程师", "email": "zhangsan@company.com", "phone": "13800138001"},
        "李四": {"name": "李四", "department": "人事部", "position": "人事经理", "email": "lisi@company.com", "phone": "13800138002"},
        "王五": {"name": "王五", "department": "财务部", "position": "财务主管", "email": "wangwu@company.com", "phone": "13800138003"},
        "赵六": {"name": "赵六", "department": "市场部", "position": "市场总监", "email": "zhaoliu@company.com", "phone": "13800138004"},
        "陈七": {"name": "陈七", "department": "技术部", "position": "前端工程师", "email": "chenqi@company.com", "phone": "13800138005"},
    }
    
    # 尝试精确匹配
    if employee_name in employees_db:
        return employees_db[employee_name]
    
    # 尝试模糊匹配
    for name, info in employees_db.items():
        if employee_name in name or name in employee_name:
            return info
    
    return {"error": f"未找到员工 '{employee_name}' 的信息"}


@mcp.tool
def check_meeting_room(
    date: Annotated[str, "日期，格式为 YYYY-MM-DD，例如 '2026-04-16'"],
    time_slot: Annotated[str, "时间段，可选值: 'morning'(上午9-12点), 'afternoon'(下午14-17点), 'evening'(晚上18-21点)"]
):
    """Checks meeting room availability and returns a list of available rooms for the specified date and time slot."""
    # 模拟会议室数据
    meeting_rooms = ["会议室A(10人)", "会议室B(20人)", "会议室C(5人)", "会议室D(50人大厅)"]
    
    # 模拟已预订情况（实际应该从数据库查询）
    booked_rooms = {
        "2026-04-16_morning": ["会议室A(10人)"],
        "2026-04-16_afternoon": ["会议室B(20人)", "会议室C(5人)"],
        "2026-04-17_morning": ["会议室D(50人大厅)"],
    }
    
    key = f"{date}_{time_slot}"
    available_rooms = [room for room in meeting_rooms if room not in booked_rooms.get(key, [])]
    
    return {
        "date": date,
        "time_slot": time_slot,
        "available_rooms": available_rooms,
        "booked_rooms": booked_rooms.get(key, []),
        "message": f"在 {date} {time_slot} 时间段，共有 {len(available_rooms)} 个会议室可用"
    }


@mcp.tool
def submit_leave_request(
    employee_name: Annotated[str, "申请人姓名"],
    leave_type: Annotated[str, "请假类型，可选值: 'annual'(年假), 'sick'(病假), 'personal'(事假), 'marriage'(婚假), 'maternity'(产假)"],
    start_date: Annotated[str, "开始日期，格式为 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期，格式为 YYYY-MM-DD"],
    reason: Annotated[str, "请假原因说明"]
):
    """Submits a leave request for an employee. Returns confirmation with leave details and approval status."""
    from datetime import datetime
    
    try:
        # 计算请假天数
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days + 1
        
        if days <= 0:
            return {"error": "结束日期必须晚于或等于开始日期"}
        
        # 请假类型映射
        leave_type_map = {
            "annual": "年假",
            "sick": "病假",
            "personal": "事假",
            "marriage": "婚假",
            "maternity": "产假"
        }
        
        leave_type_cn = leave_type_map.get(leave_type, leave_type)
        
        # 模拟提交成功（实际应该写入数据库并触发审批流程）
        result = {
            "status": "submitted",
            "request_id": f"LEAVE{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "employee_name": employee_name,
            "leave_type": leave_type_cn,
            "start_date": start_date,
            "end_date": end_date,
            "total_days": days,
            "reason": reason,
            "approval_status": "pending",
            "message": f"请假申请已提交，等待审批。申请编号: LEAVE{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        return result
        
    except ValueError as e:
        return {"error": f"日期格式错误，请使用 YYYY-MM-DD 格式。错误详情: {str(e)}"}
    except Exception as e:
        return {"error": f"提交失败: {str(e)}"}
