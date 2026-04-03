import os
import base64
import fitz  # PyMuPDF
from openai import OpenAI

# 1. 配置参数
# API Key 参考自 04_Pydantic与Tools.py
API_KEY = "sk-8b47344f618342eaa3fdbab260e9e7a1"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-plus"

# 选择一个本地 PDF 文件
pdf_path = r"H:\Desktop\AI\预科Python 【八斗学院】\第6周：RAG工程化实现\Week06\汽车知识手册.pdf"

def pdf_to_base64_image(pdf_path, page_num=0):
    """
    将 PDF 的指定页码转换为 Base64 编码的图片字符串
    """
    # 打开 PDF
    doc = fitz.open(pdf_path)
    # 选择页面
    page = doc.load_page(page_num)
    # 将页面渲染为图片 (pixmap)
    # zoom 控制清晰度，2.0 表示 2倍缩放
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # 将图片保存为字节流
    img_bytes = pix.tobytes("png")
    doc.close()
    
    # 转换为 Base64
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def parse_pdf_with_qwen():
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到 PDF 文件 {pdf_path}")
        return

    print(f"正在读取 PDF: {os.path.basename(pdf_path)}")
    print("正在将第一页转换为图片...")
    try:
        image_data_url = pdf_to_base64_image(pdf_path, page_num=0)
    except Exception as e:
        print(f"PDF 转换失败: {e}")
        return

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    print(f"正在调用云端模型 {MODEL_NAME} 进行解析...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请提取其中的关键文字和信息。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            stream=True # 建议流式输出，避免长文本超时
        )

        print("\n--- Qwen-VL 解析结果 ---")
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n" + "-"*30)

    except Exception as e:
        print(f"模型调用失败: {e}")

if __name__ == "__main__":
    parse_pdf_with_qwen()
