import base64
import io
import requests
import fitz
from PIL import Image
import re

# ================== 配置 ==================
PDF_PATH = "test.pdf"
API_KEY = "sk-bAAAdc69e65543658466dc1bcfec3233"
# =========================================

def pdf_first_page_to_image(pdf_path, dpi=150):
    """使用 PyMuPDF 将 PDF 第一页转为 PIL Image"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# 1. 转换 PDF 第一页为图片
try:
    img = pdf_first_page_to_image(PDF_PATH, dpi=150)
except FileNotFoundError:
    print(f"错误：找不到文件 '{PDF_PATH}'，请检查文件路径是否正确。")
    exit(1)

img.save("page1.jpg")
print("图片已保存为 page1.jpg，请根据图片内容修改下方的问题和选项。")

# 2. 编码图片
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
base64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

# 3. 构造视觉推理问题
question = "请根据文中内容做出判断，以下哪个选项才符合？"
options = ["A. 这是关于考试的文档", "B. 这是一份汽车驾驶文档", "C. 这是一份技术文档", "D. 这是一份关于NLP的文档"]

prompt = f"""请仔细观察图片中的场景，并进行视觉推理。
问题：{question}
选项：{' '.join(options)}
请先输出你的推理过程（详细说明为什么选择某个选项），然后输出最终答案。最终答案仅包含一个字母（例如，A）。"""

# 4. 调用 Qwen-VL
url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "model": "qwen-vl-plus",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{base64_img}"},
                    {"text": prompt}
                ]
            }
        ]
    },
    "parameters": {"result_format": "message"}
}

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    content_raw = data['output']['choices'][0]['message']['content']
    if isinstance(content_raw, list):
        # 拼接所有 text 字段（也可能包含其他类型，如图片，但这里只取文本）
        full_text = ''.join([item['text'] for item in content_raw if 'text' in item])
    else:
        full_text = content_raw

    print("\n" + "="*50)
    print("模型完整输出")
    print("="*50)
    print(full_text)
    print("="*50)

    # 提取最终答案字母
    lines = full_text.strip().split('\n')
    last_line = ""
    for line in reversed(lines):
        if line.strip():
            last_line = line.strip()
            break
    match = re.search(r'([A-Z])$', last_line)
    if match:
        answer = match.group(1)
        print(f"\n最终答案：{answer}")
    else:
        for line in lines:
            if re.match(r'^[A-Z]$', line.strip()):
                print(f"\n最终答案：{line.strip()}")
                break
        else:
            print("\n无法自动提取答案字母，请从上面输出中手动查看。")

except Exception as e:
    print("\n请求失败：", e)
    if hasattr(e, 'response') and e.response:
        print("状态码：", e.response.status_code)
        print("响应文本：", e.response.text)