from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional


def find_default_pdf(search_root: Path) -> Optional[Path]:
    preferred = search_root / "Week10-多模态大模型.pdf"
    if preferred.exists():
        return preferred

    pdf_files = sorted(search_root.glob("*.pdf"))
    return pdf_files[0] if pdf_files else None


def render_first_page(pdf_path: Path, output_image_path: Path, zoom: float) -> Path:
    import fitz

    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF 没有页面: {pdf_path}")

        page = document.load_page(0)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(str(output_image_path))

    return output_image_path


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_text_from_message_content(content) -> str:
    if isinstance(content, str):
        return content

    texts = []
    for item in content or []:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(item.get("text", ""))
    return "\n".join(part for part in texts if part)


def call_qwen_vl(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    image_data_url: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "你是一个擅长文档解析的视觉助手。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    return extract_text_from_message_content(completion.choices[0].message.content)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    course_dir = script_dir.parent
    default_pdf = find_default_pdf(course_dir)

    parser = argparse.ArgumentParser(
        description="作业2：将本地 PDF 的第一页转成图片后，调用阿里云 Qwen-VL 解析。",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=default_pdf,
        help="本地 PDF 路径。默认优先使用课程目录中的 Week10-多模态大模型.pdf。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="输出目录，会保存首页图片和模型解析结果。",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-plus",
        help="阿里云百炼视觉模型名称。",
    )
    parser.add_argument(
        "--prompt",
        default="qwenvl markdown",
        help="传给视觉模型的提示词。文档解析场景推荐 qwenvl markdown。",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="PDF 转图像时的放大倍率，默认 2.0。",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="存放 DashScope API Key 的环境变量名。",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        help="阿里云 OpenAI 兼容接口地址。",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只将 PDF 首页转成图片，不调用云端模型。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pdf_path is None:
        raise FileNotFoundError("当前课程目录下没有找到可作为默认输入的 PDF，请通过 --pdf-path 指定。")

    pdf_path = args.pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF: {pdf_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = output_dir / f"{pdf_path.stem}_page1.png"
    output_markdown_path = output_dir / f"{pdf_path.stem}_page1_qwenvl.md"

    render_first_page(pdf_path, output_image_path, args.zoom)
    print(f"PDF 首页图片已保存: {output_image_path}")

    if args.render_only:
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"未读取到环境变量 {args.api_key_env}，请先在本地设置阿里云 DashScope API Key。"
        )

    image_data_url = image_to_data_url(output_image_path)
    result = call_qwen_vl(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        image_data_url=image_data_url,
    )

    output_markdown_path.write_text(result, encoding="utf-8")
    print(f"解析结果已保存: {output_markdown_path}")
    print("解析内容预览:")
    print(result[:1000])


if __name__ == "__main__":
    main()
