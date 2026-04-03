import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# ---------- 1. 设置设备 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ---------- 2. 加载模型和处理器 ----------
model_name = "../../../models/chinese-clip-vit-base-patch16"
model = ChineseCLIPModel.from_pretrained(model_name).to(device)
processor = ChineseCLIPProcessor.from_pretrained(model_name)
print("模型加载完成！")

# ---------- 3. 加载本地图像 ----------
image_path = "./cat.jpg"  # <--- 修改这里

try:
    image = Image.open(image_path)
    print(f"成功加载图片: {image_path}")
except FileNotFoundError:
    print(f"错误：在路径 {image_path} 未找到图片。请检查文件路径。")
    exit()

# ---------- 4. 定义候选类别（零样本分类的关键）----------
# 定义任想让模型判断的类别，使用中文
classes = ["狗", "小狗", "幼犬", "猫", "鸟", "汽车", "狐狸", "狼", "泰迪熊", "金毛犬"]

# 对每个类别使用一个通用模板
texts = [f"一张{label}的照片" for label in classes]
print(f"候选类别: {texts}")

# ---------- 5. 推理 ----------
# 使用 processor 处理图像和文本，并直接传入模型
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    # outputs.logits_per_image 是一个形状为 (1, num_texts) 的张量，
    # 表示图像与每个文本描述的相似度分数
    logits_per_image = outputs.logits_per_image
    # 应用 softmax 得到概率分布
    probs = logits_per_image.softmax(dim=1)

# ---------- 6. 输出结果 ----------
print("\n=== 分类结果 ===")
# 将结果转换为 Python 列表
probs_list = probs.cpu().numpy()[0]
# 按概率从高到低排序，并打印
sorted_indices = probs_list.argsort()[::-1]

for i, idx in enumerate(sorted_indices):
    print(f"{texts[idx]} (类别: {classes[idx]}) : {probs_list[idx]:.4f}")
    if i == 2:  # 只显示前三名
        break

# 获取最佳预测
best_idx = probs_list.argmax()
print(f"\n🏆 最终预测: {classes[best_idx]} (置信度: {probs_list[best_idx]:.4f})")