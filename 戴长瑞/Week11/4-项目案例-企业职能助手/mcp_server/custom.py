import requests
from fastmcp import FastMCP

TOKEN = "738b541a5f7a"
mcp = FastMCP(
    name="Custom-MCP-Server",
    instructions="提供一些实用的自定义功能：BMI计算、股票查询、文本翻译。"
)

# ========== 工具1：计算BMI ==========
@mcp.tool
def calculate_bmi(
    weight_kg: float,
    height_m: float
) -> str:
    print(f"[TOOL CALL] calculate_bmi(weight={weight_kg}, height={height_m})")
    """
    根据体重(kg)和身高(m)计算BMI指数，并给出健康建议。
    """
    if height_m <= 0:
        return "身高必须大于0"
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        advice = "体重过轻"
    elif bmi < 24:
        advice = "正常范围"
    elif bmi < 28:
        advice = "超重"
    else:
        advice = "肥胖"
    result = f"BMI = {bmi:.1f}，{advice}。"
    print(f"[TOOL RESULT] {result}")
    return f"BMI = {bmi:.1f}，{advice}。"

# ========== 工具2：查询股票实时价格==========
@mcp.tool
def get_stock_price(symbol: str) -> str:
    print(f"[TOOL CALL] get_stock_price(symbol={symbol})")
    """
    获取指定股票代码（如 'AAPL', 'BABA'）的实时价格。
    这里使用免费API：Alpha Vantage 示例（需要注册apikey），或使用新浪财经接口。
    为演示，我们调用一个公开的模拟API。
    """
    # 示例：使用 Yahoo Finance 非官方API（仅演示，生产环境请用稳定接口）
    try:
        # 注意：你需要一个可用的股票API，下面仅作示意
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        price = float(price)
        result = f"{symbol.upper()} 当前价格: ${price}"
        print(f"[TOOL RESULT] {result}")
        return f"{symbol.upper()} 当前价格: ${price}"
    except Exception as e:
        return f"查询失败：{e}。请检查股票代码或更换API。"

# ========== 工具3：简单文本翻译（调用免费翻译API）==========
@mcp.tool
def translate_text(
    text: str,
    target_lang: str = "zh"
) -> str:
    print(f"[TOOL CALL] translate_text(text='{text}', target_lang='{target_lang}')")
    """
    将英文文本翻译为指定语言（默认中文）。
    使用 mymemory 免费翻译API。
    """
    # MyMemory 公开接口，不需要key（有每日限制）
    url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"en|{target_lang}"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        translated = data["responseData"]["translatedText"]
        print(f"[TOOL RESULT] {translated}")
        return translated
    except Exception as e:
        error_msg = f"翻译失败：{e}"
        print(f"[TOOL ERROR] {error_msg}")
        return f"翻译失败：{e}"

# if __name__ == "__main__":
#     # 运行 SSE 服务（默认端口 8900 可能会冲突，建议换一个端口，比如 8901）
#     mcp.run(transport="sse", host="0.0.0.0", port=8901)