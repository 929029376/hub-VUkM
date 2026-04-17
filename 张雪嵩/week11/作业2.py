import re
import random
import string
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)



@mcp.tool
def get_employee_info(employee_name: Annotated[str, "员工姓名"]):
    """根据员工姓名查询员工基本信息，包括部门、职位、入职日期等。"""
    
    employee_database = {
        "张三": {
            "姓名": "张三",
            "部门": "技术研发部",
            "职位": "高级工程师",
            "入职日期": "2020-03-15",
            "邮箱": "zhangsan@company.com",
            "工号": "EMP001"
        },
        "李四": {
            "姓名": "李四",
            "部门": "市场营销部",
            "职位": "市场经理",
            "入职日期": "2019-06-20",
            "邮箱": "lisi@company.com",
            "工号": "EMP002"
        },
        "王五": {
            "姓名": "王五",
            "部门": "人力资源部",
            "职位": "HR专员",
            "入职日期": "2021-01-10",
            "邮箱": "wangwu@company.com",
            "工号": "EMP003"
        },
        "赵六": {
            "姓名": "赵六",
            "部门": "财务部",
            "职位": "财务主管",
            "入职日期": "2018-09-05",
            "邮箱": "zhaoliu@company.com",
            "工号": "EMP004"
        },
        "陈七": {
            "姓名": "陈七",
            "部门": "产品设计部",
            "职位": "产品经理",
            "入职日期": "2022-02-28",
            "邮箱": "chenqi@company.com",
            "工号": "EMP005"
        }
    }
    
    if employee_name in employee_database:
        info = employee_database[employee_name]
        return {
            "status": "success",
            "message": f"查询到员工 {employee_name} 的信息",
            "data": info
        }
    else:
        return {
            "status": "not_found",
            "message": f"未找到员工 {employee_name} 的信息，请确认姓名是否正确",
            "available_employees": list(employee_database.keys())
        }


@mcp.tool
def calculator(expression: Annotated[str, "数学表达式，如 '2+3*4' 或 '100/5'"]):
    """执行基本数学运算，支持加减乘除和括号运算。"""
    
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return {
                "status": "error",
                "message": "表达式包含不允许的字符，只支持数字、加减乘除和括号"
            }
        
        result = eval(expression)
        
        return {
            "status": "success",
            "expression": expression,
            "result": result,
            "message": f"计算结果: {expression} = {result}"
        }
    except ZeroDivisionError:
        return {
            "status": "error",
            "message": "错误：除数不能为零"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"计算错误：{str(e)}"
        }


@mcp.tool
def generate_password(length: Annotated[int, "密码长度，默认12位"] = 12):
    """生成指定长度的随机安全密码，包含大小写字母、数字和特殊字符。"""
    
    if length < 6:
        return {
            "status": "error",
            "message": "密码长度至少为6位"
        }
    
    if length > 32:
        return {
            "status": "error",
            "message": "密码长度不能超过32位"
        }
    
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = "!@#$%^&*"
    
    password_chars = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special)
    ]
    
    all_chars = lowercase + uppercase + digits + special
    password_chars.extend(random.choices(all_chars, k=length - 4))
    
    random.shuffle(password_chars)
    password = ''.join(password_chars)
    
    strength = "强" if length >= 12 else "中" if length >= 8 else "弱"
    
    return {
        "status": "success",
        "password": password,
        "length": length,
        "strength": strength,
        "message": f"已生成{strength}强度密码（{length}位）: {password}"
    }
