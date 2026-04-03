from openai import OpenAI
import os
import base64
from pathlib import Path
from typing import Tuple, Optional
import fitz  # PyMuPDF


# ---------------------- 配置 ----------------------
class Config:
    """配置管理类"""
    # 模型配置
    MODEL_NAME = "xxx"  # 使用视觉模型，原qwen3.5-plus不支持图片
    BASE_URL = "xxxxxxx"

    # PDF配置
    PDF_DPI = 2.0  # 图片清晰度倍数
    IMAGE_FORMAT = "png"

    # 提示词
    PROMPT = "请详细解析这一页PDF的内容，包括文字、题目、图表、结构等"

    # 环境变量
    API_KEY_ENV = "DASHSCOPE_API_KEY"


class PDFProcessor:
    """PDF处理类"""

    @staticmethod
    def first_page_to_base64(pdf_path: str, zoom: float = 2.0) -> Optional[str]:
        """
        将PDF第一页转换为base64编码的图片

        Args:
            pdf_path: PDF文件路径
            zoom: 缩放倍数（清晰度）

        Returns:
            base64编码的图片URL，失败返回None
        """
        pdf_path = Path(pdf_path)

        # 检查文件是否存在
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"文件不是PDF格式: {pdf_path}")

        doc = None
        try:
            # 打开PDF文档
            doc = fitz.open(pdf_path)

            if len(doc) == 0:
                raise ValueError("PDF文件没有页面")

            # 获取第一页
            page = doc[0]

            # 转换为图片（提高清晰度）
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)

            # 转换为base64
            img_bytes = pix.tobytes(Config.IMAGE_FORMAT)
            base64_str = base64.b64encode(img_bytes).decode("utf-8")

            return f"data:image/{Config.IMAGE_FORMAT};base64,{base64_str}"

        except Exception as e:
            print(f"❌ PDF处理失败: {e}")
            raise
        finally:
            if doc:
                doc.close()


class StreamProcessor:
    """流式输出处理器"""

    def __init__(self):
        self.reasoning_content = ""
        self.answer_content = ""
        self.is_answering = False

    def process_chunk(self, chunk) -> Tuple[str, str]:
        """
        处理单个响应块

        Returns:
            (reasoning_part, answer_part) 新增的内容
        """
        reasoning_part = ""
        answer_part = ""

        if not chunk.choices:
            # 打印token使用情况
            if hasattr(chunk, 'usage') and chunk.usage:
                print("\n📊 Token使用统计:")
                print(f"   - Prompt tokens: {chunk.usage.prompt_tokens}")
                print(f"   - Completion tokens: {chunk.usage.completion_tokens}")
                print(f"   - Total tokens: {chunk.usage.total_tokens}")
            return reasoning_part, answer_part

        delta = chunk.choices[0].delta

        # 处理思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            content = delta.reasoning_content
            reasoning_part = content
            self.reasoning_content += content
            print(content, end='', flush=True)

        # 处理答案内容
        elif delta.content:
            if not self.is_answering:
                print("\n" + "=" * 20 + "💡 完整回复" + "=" * 20 + "\n")
                self.is_answering = True

            content = delta.content
            answer_part = content
            self.answer_content += content
            print(content, end='', flush=True)

        return reasoning_part, answer_part


def validate_api_key() -> str:
    """验证API密钥"""
    api_key = os.getenv(Config.API_KEY_ENV)

    if not api_key:
        raise EnvironmentError(
            f"❌ 环境变量 {Config.API_KEY_ENV} 未设置\n"
            f"请运行: export {Config.API_KEY_ENV}='your-api-key'"
        )

    return api_key


def main():
    """主函数"""
    print("🚀 启动PDF解析程序")
    print(f"📄 目标文件: {PDF_FILE_PATH}")
    print("-" * 50)

    try:
        # 1. 验证API密钥
        api_key = validate_api_key()

        # 2. 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url=Config.BASE_URL
        )

        # 3. 处理PDF
        print("🔄 正在处理PDF文件...")
        pdf_image_base64 = PDFProcessor.first_page_to_base64(
            PDF_FILE_PATH,
            zoom=Config.PDF_DPI
        )
        print("✅ PDF转换完成")

        # 4. 创建请求
        print("🤖 正在调用模型API...")
        completion = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": pdf_image_base64}},
                        {"type": "text", "text": Config.PROMPT}
                    ]
                }
            ],
            stream=True,
            temperature=0.7,  # 添加温度参数控制创造性
            max_tokens=2000,  # 限制最大输出长度
        )

        # 5. 处理流式输出
        print("\n" + "=" * 20 + "🤔 思考过程" + "=" * 20 + "\n")
        processor = StreamProcessor()

        for chunk in completion:
            processor.process_chunk(chunk)

        print("\n" + "=" * 50)
        print("✅ 解析完成")

        # 6. 可选：保存结果到文件
        save_results = input("\n💾 是否保存结果到文件？(y/n): ").lower().strip()
        if save_results == 'y':
            output_file = Path(PDF_FILE_PATH).stem + "_analysis.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("PDF解析结果\n")
                f.write("=" * 50 + "\n\n")
                f.write("【思考过程】\n")
                f.write(processor.reasoning_content + "\n\n")
                f.write("【完整回复】\n")
                f.write(processor.answer_content + "\n")
            print(f"✅ 结果已保存到: {output_file}")

    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
    except EnvironmentError as e:
        print(f"❌ 配置错误: {e}")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置PDF路径（可以改为其他路径）
    PDF_FILE_PATH = "xxx"

    main()