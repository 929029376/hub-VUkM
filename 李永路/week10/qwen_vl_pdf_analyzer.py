"""
使用阿里云 Qwen-VL 大模型解析 PDF 文件的特定页面
本脚本用于解析：应阔浩 -2025 自如企业级 AI 架构落地的思考与实践.pdf 的第 2 页
"""

import base64
from openai import OpenAI
import fitz  # PyMuPDF, 用于处理 PDF 文件
from PIL import Image
import io
import os

# 配置阿里云 API
API_KEY = ""  # 
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def pdf_page_to_image(pdf_path, page_num):
    """
    将 PDF 的指定页面转换为图片
    
    Args:
        pdf_path: PDF 文件路径
        page_num: 页码 (从 1 开始)
    
    Returns:
        PIL Image 对象
    """
    # 打开 PDF 文件
    doc = fitz.open(pdf_path)
    
    # 检查页码是否有效
    if page_num < 1 or page_num > len(doc):
        raise ValueError(f"页码超出范围！PDF 共有 {len(doc)} 页")
    
    # 获取指定页面
    page = doc[page_num - 1]
    
    # 设置渲染参数（分辨率越高，图片越清晰）
    zoom = 2.0  # 缩放比例
    mat = fitz.Matrix(zoom, zoom)
    
    # 将页面渲染为图片
    pix = page.get_pixmap(matrix=mat)
    
    # 转换为 PNG 格式
    img_data = pix.tobytes("png")
    
    # 使用 PIL 打开图片
    image = Image.open(io.BytesIO(img_data))
    
    # 关闭 PDF 文档
    doc.close()
    
    return image

def image_to_base64(image, format="PNG"):
    """
    将 PIL Image 转换为 base64 编码
    
    Args:
        image: PIL Image 对象
        format: 图片格式
    
    Returns:
        base64 字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_pdf_page_with_qwen(image_path, prompt="请详细描述这张图片的内容"):
    """
    使用 Qwen-VL 分析 PDF 页面
    
    Args:
        image_path: 图片路径或 base64 字符串
        prompt: 提示词
    
    Returns:
        Qwen-VL 的响应文本
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=ENDPOINT,
    )
    
    # 判断是文件路径还是 base64
    if os.path.exists(image_path):
        # 如果是文件路径，读取并转换为 base64
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
    else:
        # 如果已经是 base64
        image_base64 = image_path
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # 调用 Qwen-VL模型
    response = client.chat.completions.create(
        model="qwen-vl-max",  # 或 qwen-vl-plus, qwen-vl-max-latest
        messages=messages,
        stream=False
    )
    
    return response.choices[0].message.content

def main():
    """主函数"""
    # PDF 文件路径
    pdf_path = "应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"
    
    # 要解析的页码
    page_num = 2
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误：找不到文件 {pdf_path}")
        return
    
    print(f"正在加载 PDF 文件：{pdf_path}")
    print(f"正在提取第 {page_num} 页...")
    
    try:
        # 将 PDF 页面转换为图片
        image = pdf_page_to_image(pdf_path, page_num)
        
        # 保存临时图片（可选）
        temp_image_path = f"temp_page_{page_num}.png"
        image.save(temp_image_path)
        print(f"已将第 {page_num} 页保存为图片：{temp_image_path}")
        
        # 定义分析提示词
        prompt = """请详细分析这张 PDF 页面的内容，包括：
1. 页面的标题和主要内容
2. 关键的技术架构或概念
3. 图表中的信息（如果有）
4. 重要的数据或结论
请用中文回答。"""
        
        print("\n正在调用 Qwen-VL 进行分析...")
        
        # 使用 Qwen-VL 分析图片
        result = analyze_pdf_page_with_qwen(temp_image_path, prompt)
        
        print("\n" + "="*60)
        print("Qwen-VL 分析结果:")
        print("="*60)
        print(result)
        print("="*60)
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"\n已清理临时文件：{temp_image_path}")
        
    except Exception as e:
        print(f"发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
