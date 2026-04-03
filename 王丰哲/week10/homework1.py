from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from PIL import Image, ImageDraw


DEFAULT_LABELS = ["小狗", "小猫", "兔子", "汽车", "蛋糕", "风景"]


def create_cartoon_dog_sample(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 768, 768
    image = Image.new("RGB", (width, height), "#d8efff")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 520, width, height), fill="#8dd18b")
    draw.ellipse((60, 50, 708, 700), fill="#d8a96f")
    draw.ellipse((120, 160, 648, 680), fill="#e7b97b")

    draw.ellipse((40, 120, 250, 410), fill="#a66d3b")
    draw.ellipse((518, 120, 728, 410), fill="#a66d3b")
    draw.ellipse((95, 175, 230, 390), fill="#7c4e26")
    draw.ellipse((538, 175, 673, 390), fill="#7c4e26")

    draw.ellipse((185, 250, 315, 380), fill="white")
    draw.ellipse((453, 250, 583, 380), fill="white")
    draw.ellipse((225, 290, 275, 340), fill="#2c2118")
    draw.ellipse((493, 290, 543, 340), fill="#2c2118")
    draw.ellipse((240, 305, 258, 323), fill="white")
    draw.ellipse((508, 305, 526, 323), fill="white")

    draw.ellipse((235, 410, 533, 635), fill="#f5deb9")
    draw.ellipse((315, 430, 455, 535), fill="#5b3220")
    draw.polygon([(385, 500), (360, 545), (410, 545)], fill="#3f2115")

    draw.line((320, 560, 370, 590), fill="#3f2115", width=8)
    draw.line((450, 560, 400, 590), fill="#3f2115", width=8)
    draw.arc((325, 560, 385, 610), start=15, end=155, fill="#e27a8d", width=7)
    draw.arc((385, 560, 445, 610), start=25, end=165, fill="#e27a8d", width=7)

    draw.ellipse((205, 210, 285, 280), outline="#70422b", width=8)
    draw.ellipse((483, 210, 563, 280), outline="#70422b", width=8)
    draw.ellipse((90, 90, 165, 155), fill="white")
    draw.ellipse((600, 110, 660, 165), fill="white")

    image.save(output_path)
    return output_path


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_prompts(labels: Sequence[str], prompt_template: str) -> List[str]:
    if "{label}" in prompt_template:
        return [prompt_template.format(label=label) for label in labels]
    return [prompt_template.format(label) for label in labels]


def classify_image(
    image_path: Path,
    labels: Sequence[str],
    model_name_or_path: str,
    prompt_template: str,
    device: str,
) -> List[tuple[str, float]]:
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    resolved_device = resolve_device(device)
    prompts = build_prompts(labels, prompt_template)

    image = Image.open(image_path).convert("RGB")
    processor = ChineseCLIPProcessor.from_pretrained(model_name_or_path)
    model = ChineseCLIPModel.from_pretrained(model_name_or_path).to(resolved_device)
    model.eval()

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(resolved_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()

    results = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    return results


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="作业1：使用本地小狗图片进行 Chinese-CLIP zero-shot classification。",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=script_dir / "dog_sample.png",
        help="本地图片路径。默认使用脚本目录下的 dog_sample.png。",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="候选类别，多个类别用空格分隔。",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="OFA-Sys/chinese-clip-vit-base-patch16",
        help="Hugging Face 模型名，或你本地课程中的 Chinese-CLIP 模型目录。",
    )
    parser.add_argument(
        "--prompt-template",
        default="这是一张{label}的照片",
        help="文本提示模板，支持 {label} 占位符。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="运行设备，默认自动选择。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="展示概率最高的前 K 个类别。",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="只生成本地小狗样例图，不执行分类。",
    )
    parser.add_argument(
        "--create-sample-if-missing",
        dest="create_sample_if_missing",
        action="store_true",
        default=True,
        help="若图片不存在，则自动生成一张小狗样例图。",
    )
    parser.add_argument(
        "--no-create-sample-if-missing",
        dest="create_sample_if_missing",
        action="store_false",
        help="若图片不存在则直接报错。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image_path.resolve()

    if not image_path.exists():
        if not args.create_sample_if_missing:
            raise FileNotFoundError(f"找不到图片: {image_path}")
        created_path = create_cartoon_dog_sample(image_path)
        print(f"已生成本地小狗样例图: {created_path}")

    if args.create_only:
        print(f"样例图片已准备完成: {image_path}")
        return

    results = classify_image(
        image_path=image_path,
        labels=args.labels,
        model_name_or_path=args.model_name_or_path,
        prompt_template=args.prompt_template,
        device=args.device,
    )

    print(f"图片路径: {image_path}")
    print(f"候选类别: {', '.join(args.labels)}")
    print("分类结果:")
    for label, score in results[: args.top_k]:
        print(f"- {label}: {score:.4%}")


if __name__ == "__main__":
    main()
