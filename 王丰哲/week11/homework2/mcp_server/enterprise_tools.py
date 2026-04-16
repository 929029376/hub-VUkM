# [新建文件] 自定义企业工具模块
from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

mcp = FastMCP(
    name="Enterprise-Tools-MCP-Server",
    instructions="""This server contains custom enterprise workflow tools.""",
)


EMPLOYEE_PROFILES = {
    "张三": {
        "department": "研发部",
        "title": "后端工程师",
        "annual_leave_remaining": 6,
        "comp_leave_remaining_hours": 12,
        "payday": "每月10日",
        "last_payroll_month": "2026-03",
        "last_payroll_amount": 18500,
    },
    "李四": {
        "department": "市场部",
        "title": "市场运营",
        "annual_leave_remaining": 9,
        "comp_leave_remaining_hours": 4,
        "payday": "每月10日",
        "last_payroll_month": "2026-03",
        "last_payroll_amount": 13200,
    },
    "王敏": {
        "department": "人事行政部",
        "title": "HRBP",
        "annual_leave_remaining": 11,
        "comp_leave_remaining_hours": 0,
        "payday": "每月10日",
        "last_payroll_month": "2026-03",
        "last_payroll_amount": 15600,
    },
}

MEETING_ROOMS = {
    "A101": {"capacity": 8, "equipment": ["投影仪", "白板"]},
    "B201": {"capacity": 12, "equipment": ["电视屏幕", "视频会议终端"]},
    "C305": {"capacity": 4, "equipment": ["白板"]},
}

MEETING_BOOKINGS = {
    ("A101", "2026-04-17", "14:00", "15:00"): {
        "requester": "王敏",
        "topic": "月度复盘",
    },
    ("B201", "2026-04-18", "10:00", "11:00"): {
        "requester": "张三",
        "topic": "需求评审",
    },
}


def _get_employee_profile(employee_name: str) -> dict:
    return EMPLOYEE_PROFILES.get(
        employee_name,
        {
            "department": "未知部门",
            "title": "未登记岗位",
            "annual_leave_remaining": 5,
            "comp_leave_remaining_hours": 0,
            "payday": "每月10日",
            "last_payroll_month": "2026-03",
            "last_payroll_amount": 12000,
        },
    )


@mcp.tool
def query_employee_leave_balance(
    employee_name: Annotated[str, "Employee full name used inside the company directory."]
):
    """Query the remaining annual leave and compensatory leave balance for an employee."""
    profile = _get_employee_profile(employee_name)
    return {
        "employee_name": employee_name,
        "department": profile["department"],
        "title": profile["title"],
        "annual_leave_remaining_days": profile["annual_leave_remaining"],
        "comp_leave_remaining_hours": profile["comp_leave_remaining_hours"],
        "message": f"{employee_name} 当前剩余年假 {profile['annual_leave_remaining']} 天，调休 {profile['comp_leave_remaining_hours']} 小时。",
    }


@mcp.tool
def query_employee_payroll(
    employee_name: Annotated[str, "Employee full name used inside the company directory."],
    month: Annotated[str, "Payroll month in YYYY-MM format, such as 2026-03."] = "2026-03",
):
    """Query the payroll schedule and payroll summary for an employee."""
    profile = _get_employee_profile(employee_name)
    is_latest_month = month == profile["last_payroll_month"]
    payroll_amount = profile["last_payroll_amount"] if is_latest_month else round(profile["last_payroll_amount"] * 0.96, 2)
    return {
        "employee_name": employee_name,
        "month": month,
        "department": profile["department"],
        "payday_rule": profile["payday"],
        "estimated_payroll_amount": payroll_amount,
        "status": "已发放" if month <= profile["last_payroll_month"] else "待发放",
        "message": f"{employee_name} 在 {month} 的工资查询结果已返回，发薪规则为 {profile['payday']}。",
    }


@mcp.tool
def book_meeting_room(
    employee_name: Annotated[str, "Requester full name."],
    room_name: Annotated[str, "Meeting room name, such as A101 or B201."],
    meeting_date: Annotated[str, "Meeting date in YYYY-MM-DD format."],
    start_time: Annotated[str, "Meeting start time in HH:MM format."],
    end_time: Annotated[str, "Meeting end time in HH:MM format."],
    topic: Annotated[str, "Meeting topic or purpose."],
):
    """Book a meeting room for an employee if the requested time slot is available."""
    if room_name not in MEETING_ROOMS:
        return {
            "success": False,
            "message": f"会议室 {room_name} 不存在，可选会议室有：{', '.join(MEETING_ROOMS.keys())}。",
        }

    booking_key = (room_name, meeting_date, start_time, end_time)
    if booking_key in MEETING_BOOKINGS:
        current_booking = MEETING_BOOKINGS[booking_key]
        return {
            "success": False,
            "room_name": room_name,
            "meeting_date": meeting_date,
            "time_range": f"{start_time}-{end_time}",
            "conflict_with": current_booking,
            "suggestion": "可以尝试改订 C305，或调整到相邻空闲时间段。",
            "message": f"{room_name} 在 {meeting_date} {start_time}-{end_time} 已被占用。",
        }

    MEETING_BOOKINGS[booking_key] = {"requester": employee_name, "topic": topic}
    confirmation_id = f"MR-{meeting_date.replace('-', '')}-{room_name}-{start_time.replace(':', '')}"
    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "room_name": room_name,
        "meeting_date": meeting_date,
        "time_range": f"{start_time}-{end_time}",
        "requester": employee_name,
        "topic": topic,
        "room_capacity": MEETING_ROOMS[room_name]["capacity"],
        "equipment": MEETING_ROOMS[room_name]["equipment"],
        "message": f"已成功为 {employee_name} 预订 {room_name}，时间为 {meeting_date} {start_time}-{end_time}。",
    }
