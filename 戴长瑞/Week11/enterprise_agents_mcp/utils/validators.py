import re

def validate_input(text: str) -> dict:
    """返回验证结果和错误信息"""
    if not text or not text.strip():
        return {"valid": False, "error": "输入不能为空"}
    if len(text) > 500:
        return {"valid": False, "error": "输入长度不能超过500字符"}
    if re.search(r'[<>{}]', text):
        return {"valid": False, "error": "输入包含非法字符"}
    return {"valid": True, "error": None}