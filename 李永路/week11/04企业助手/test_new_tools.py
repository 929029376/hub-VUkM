"""
测试新增的3个自定义工具
"""
import asyncio
from fastmcp import Client
from mcp_server.tool import mcp as tool_mcp
from mcp_server.mcp_server_main import mcp

async def test_new_tools():
    """测试新添加的三个工具"""
    
    # 设置MCP服务器
    await mcp.import_server(tool_mcp, prefix="")
    
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("=" * 80)
        print("所有可用工具:")
        print([t.name for t in tools])
        print("=" * 80)
        
        # 测试1: 查询员工信息
        print("\n【测试1】查询员工信息 - 张三")
        print("-" * 80)
        result = await client.call_tool("get_employee_info", {"employee_name": "张三"})
        print(f"结果: {result}")
        
        print("\n【测试2】查询员工信息 - 李四")
        print("-" * 80)
        result = await client.call_tool("get_employee_info", {"employee_name": "李四"})
        print(f"结果: {result}")
        
        print("\n【测试3】查询不存在的员工")
        print("-" * 80)
        result = await client.call_tool("get_employee_info", {"employee_name": "不存在的人"})
        print(f"结果: {result}")
        
        # 测试2: 查询会议室
        print("\n【测试4】查询会议室可用性 - 2026-04-16 上午")
        print("-" * 80)
        result = await client.call_tool("check_meeting_room", {
            "date": "2026-04-16",
            "time_slot": "morning"
        })
        print(f"结果: {result}")
        
        print("\n【测试5】查询会议室可用性 - 2026-04-16 下午")
        print("-" * 80)
        result = await client.call_tool("check_meeting_room", {
            "date": "2026-04-16",
            "time_slot": "afternoon"
        })
        print(f"结果: {result}")
        
        print("\n【测试6】查询会议室可用性 - 2026-04-17 上午")
        print("-" * 80)
        result = await client.call_tool("check_meeting_room", {
            "date": "2026-04-17",
            "time_slot": "morning"
        })
        print(f"结果: {result}")
        
        # 测试3: 提交请假申请
        print("\n【测试7】提交年假申请")
        print("-" * 80)
        result = await client.call_tool("submit_leave_request", {
            "employee_name": "张三",
            "leave_type": "annual",
            "start_date": "2026-04-20",
            "end_date": "2026-04-22",
            "reason": "家里有事需要处理"
        })
        print(f"结果: {result}")
        
        print("\n【测试8】提交病假申请")
        print("-" * 80)
        result = await client.call_tool("submit_leave_request", {
            "employee_name": "李四",
            "leave_type": "sick",
            "start_date": "2026-04-18",
            "end_date": "2026-04-18",
            "reason": "感冒发烧需要休息"
        })
        print(f"结果: {result}")
        
        print("\n【测试9】测试错误的日期格式")
        print("-" * 80)
        result = await client.call_tool("submit_leave_request", {
            "employee_name": "王五",
            "leave_type": "personal",
            "start_date": "2026/04/20",  # 错误格式
            "end_date": "2026/04/22",
            "reason": "个人事务"
        })
        print(f"结果: {result}")
        
        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_new_tools())
