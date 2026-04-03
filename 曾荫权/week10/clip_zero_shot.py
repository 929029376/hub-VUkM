import os
import sys
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import glob

def test_all_images():
    # 设置模型和路径
    model_id = "OFA-Sys/chinese-clip-vit-base-patch16"
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
    
    # 获取目录下所有的 jpg 图片
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_paths.sort()
    
    if not image_paths:
        print(f"错误: 在 {image_dir} 目录下没有找到任何图片。")
        return

    print(f"找到 {len(image_paths)} 张图片，准备进行分类。")
    
    # 加载模型
    print(f"正在加载模型 {model_id}...")
    try:
        model = ChineseCLIPModel.from_pretrained(model_id)
        processor = ChineseCLIPProcessor.from_pretrained(model_id)
        print("模型加载成功。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 设置分类标签
    texts = ["一只狗", "一只猫", "一辆车", "建筑", "人"]
    print(f"分类标签: {texts}")

    # 循环处理每张图片
    for img_path in image_paths:
        print(f"\n" + "="*40)
        print(f"正在处理图片: {os.path.basename(img_path)}")
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 预处理和推理
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                # 获取图片与文本的相似度得分并转为概率
                probs = outputs.logits_per_image.softmax(dim=1)
            
            # 输出该图片的分类结果
            print("--- 分类结果 ---")
            results = []
            for i, text in enumerate(texts):
                prob = probs[0][i].item()
                results.append((text, prob))
                print(f"{text:<10}: {prob:.4f}")
            
            # 找到概率最高的标签
            best_match = max(results, key=lambda x: x[1])
            print(f"\n预测结果: {best_match[0]} (置信度: {best_match[1]:.4f})")
            
        except Exception as e:
            print(f"处理图片 {os.path.basename(img_path)} 时出错: {e}")

if __name__ == "__main__":
    test_all_images()
