# mcp_server/my_custom_tools.py
# 自定义企业职能工具 —— 新增的 3 个 Tool

from typing import Annotated
from datetime import datetime, date
from fastmcp import FastMCP

mcp = FastMCP(
    name="Custom-MCP-Server",
    instructions="""This server contains custom enterprise tools.""",
)


# ────────────────────────────────────────────────
# Tool 1：查询员工年假余额
# 触发词示例："我还有多少天年假？" / "查一下张三的年假"
# ────────────────────────────────────────────────
@mcp.tool
def query_annual_leave(employee_name: Annotated[str, "员工姓名"]):
    """
    Query the remaining annual leave days for an employee by name.
    Use this tool when the user asks about annual leave, vacation days, or remaining holidays.
    """
    # TODO: 替换为真实数据库查询
    mock_data = {
        "张三": 8,
        "李四": 5,
        "王五": 12,
        "赵六": 0,
    }
    days = mock_data.get(employee_name, None)
    if days is None:
        return f"未找到员工「{employee_name}」的年假信息，请确认姓名是否正确。"
    return f"员工「{employee_name}」剩余年假：{days} 天"


# ────────────────────────────────────────────────
# Tool 2：查询会议室空闲状态
# 触发词示例："明天下午3点301会议室有人吗？" / "查一下大会议室今天下午是否空闲"
# ────────────────────────────────────────────────
@mcp.tool
def query_meeting_room(
        room_name: Annotated[str, "会议室名称，例如：301会议室、大会议室"],
        date_str: Annotated[str, "查询日期，格式 YYYY-MM-DD，例如：2026-04-16"],
        time_slot: Annotated[str, "时间段，例如：上午、下午、14:00-16:00"],
):
    """
    Query whether a specific meeting room is available on a given date and time slot.
    Use this tool when the user asks about meeting room availability or wants to book a room.
    """
    # TODO: 替换为真实会议室预订系统查询
    mock_bookings = {
        ("301会议室", "2026-04-16", "下午"): "已被研发部预订（13:00-17:00）",
        ("大会议室", "2026-04-16", "上午"): "空闲",
        ("301会议室", "2026-04-17", "上午"): "空闲",
    }
    key = (room_name, date_str, time_slot)
    status = mock_bookings.get(key, "空闲")
    return f"【{room_name}】{date_str} {time_slot}：{status}"


# ────────────────────────────────────────────────
# Tool 3：计算两个日期之间的工作日天数（排除周末）
# 触发词示例："从4月1日到4月30日有多少个工作日？" / "这个月剩多少工作日"
# ────────────────────────────────────────────────
@mcp.tool
def calculate_workdays(
        start_date: Annotated[str, "开始日期，格式 YYYY-MM-DD，例如：2026-04-01"],
        end_date: Annotated[str, "结束日期，格式 YYYY-MM-DD，例如：2026-04-30"],
):
    """
    Calculate the number of working days (excluding weekends) between two dates.
    Use this tool when the user asks about workdays, business days, or deadline calculations.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            return "开始日期不能晚于结束日期"

        count = 0
        current = start
        while current <= end:
            if current.weekday() < 5:  # 0=周一 ... 4=周五
                count += 1
            current = date.fromordinal(current.toordinal() + 1)

        return f"{start_date} 到 {end_date} 共有 {count} 个工作日（不含周末）"
    except ValueError:
        return "日期格式错误，请使用 YYYY-MM-DD 格式，例如：2026-04-01"
