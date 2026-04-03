import base64
from pathlib import Path
from openai import OpenAI
import fitz  # PyMuPDF
# 将 PDF 第一页转换为图片（Base64）
from pdf2image import convert_from_path
def pdf_first_page_to_base64(pdf_path: str, dpi: int = 200) -> str:
    """
    使用 PyMuPDF 将 PDF 第一页转换为 Base64 编码的 JPEG 图片
    """
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            raise ValueError("PDF 没有页面")
        # 获取第一页
        page = doc[0]
        # 设置缩放比例（dpi / 72 是标准转换系数，dpi=200 时缩放约 2.78）
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace="rgb")
        # 转换为 JPEG 字节流
        img_bytes = pix.tobytes("jpeg")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"PDF 转换失败: {e}")
        exit(1)


def call_qwen_vl(image_base64: str, prompt: str, api_key: str, model: str = "qwen3-vl-plus", enable_thinking: bool = False):
    """
    调用阿里云百炼 Qwen-VL 模型
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 构建消息（图像使用 data URL 格式）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # 如果启用思考模式，设置参数
    extra_body = {}
    if enable_thinking:
        extra_body["enable_thinking"] = True
        # 可选：限制思考 token 数，默认 81920
        # extra_body["thinking_budget"] = 81920

    print("\n" + "=" * 20 + "模型响应" + "=" * 20 + "\n")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,  # 设为 False 简化输出，如需流式可改为 True
            **extra_body
        )
        response = completion.choices[0].message.content
        print(response)

        # 如果启用了思考模式，可以额外打印思考内容
        if enable_thinking and hasattr(completion.choices[0].message, 'reasoning_content'):
            reasoning = completion.choices[0].message.reasoning_content
            if reasoning:
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
                print(reasoning)

    except Exception as e:
        print(f"调用 API 失败: {e}")
        exit(1)


def main():
    api_key = "sk-6a38ccb6fc234175887710e6324ecaf2"
    pdf_path = "./关于共享流程讨论0331.pdf"
    modlestr = "qwen3-vl-plus"
    # 检查 PDF 文件是否存在
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"错误：文件不存在 - {pdf_path}")
        exit(1)

    print(f"正在处理 PDF: {pdf_path}")
    print(f"模型: {modlestr}, 思考模式: 开启")

    # 转换 PDF 第一页为 Base64 图片
    image_base64 = pdf_first_page_to_base64(str(pdf_path))

    # 调用模型
    call_qwen_vl(
        image_base64=image_base64,
        api_key=api_key,
        model=modlestr,
        prompt=""
    )


if __name__ == "__main__":
    main()