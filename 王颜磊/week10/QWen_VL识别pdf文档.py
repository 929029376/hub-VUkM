import os
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
from openai import OpenAI


def pdf_to_images(pdf_path):
    """将PDF转换为图像列表"""
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # 转换为高分辨率图像（200 DPI）
        mat = fitz.Matrix(2.0, 2.0)  # 2.0 = 144/72 (200 DPI ≈ 2x)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    doc.close()
    return images


def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def main():
    # 获取用户输入
    pdf_path = input("请输入PDF文件路径: ").strip()

    # 验证文件是否存在
    if not os.path.exists(pdf_path):
        print("文件不存在！请检查路径。")
        return

    api_key = input("请输入阿里云API Key: ").strip()

    # 初始化Qwen-VL客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 转换PDF为图像
    print("正在处理PDF文件...")
    try:
        images = pdf_to_images(pdf_path)
        print(f" PDF处理完成！共 {len(images)} 页")
    except Exception as e:
        print(f" PDF处理失败: {e}")
        return

    # 将所有页面转换为base64
    print(" 正在准备图像数据...")
    base64_images = [image_to_base64(img) for img in images]

    # 主问答循环
    print("\n 现在可以开始提问了！输入 'quit' 退出程序。")
    while True:
        question = input("\n 你的问题: ").strip()

        if question.lower() == 'quit':
            print(" 再见！")
            break

        if not question:
            print("请输入有效的问题。")
            continue

        # 构建多图像消息
        content = [{"type": "text", "text": question}]
        for i, base64_img in enumerate(base64_images):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })

        try:
            print("正在思考中...")
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": content}],
                max_tokens=100,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            print(f"\n💡 回答: {answer}")

        except Exception as e:
            print(f"回答失败: {e}")


if __name__ == "__main__":
    main()
