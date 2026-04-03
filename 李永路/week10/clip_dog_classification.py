import torch
from PIL import Image
import clip

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")

# 加载预训练的 CLIP模型（使用 ViT-B/32）
print("正在加载 CLIP模型...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 加载并预处理图像
print("\n正在处理图像 dog.jpg...")
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# 定义分类的文本标签（zero-shot classification）
# 你可以根据需要修改这些类别
text_labels = [
    "a photo of a dog",
    "a photo of a cat", 
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person",
    "a photo of a house",
    "a photo of a tree",
    "a photo of food",
    "a photo of a golden retriever",
    "a photo of a german shepherd",
    "a photo of a puppy"
]

print(f"\n分类标签：{len(text_labels)} 个类别")
for i, label in enumerate(text_labels, 1):
    print(f"{i}. {label}")

# 预处理文本
text = clip.tokenize(text_labels).to(device)

# 进行推理（不计算梯度）
print("\n正在进行 zero-shot 分类...")
with torch.no_grad():
    # 获取图像特征
    image_features = model.encode_image(image)
    
    # 获取文本特征
    text_features = model.encode_text(text)
    
    # 计算图像和文本的相似度
    logits_per_image, logits_per_text = model(image, text)
    
    # 使用 softmax 转换为概率
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 输出结果
print("\n" + "="*60)
print("预测结果:")
print("="*60)
for i, (label, prob) in enumerate(zip(text_labels, probs[0])):
    print(f"{i+1:2d}. {label:40s} : {prob:.4f} ({prob*100:.2f}%)")

# 找出最可能的类别
best_match_idx = probs.argmax()
print("\n" + "="*60)
print(f"最匹配的类别：{text_labels[best_match_idx]}")
print(f"置信度：{probs[0][best_match_idx]:.4f} ({probs[0][best_match_idx]*100:.2f}%)")
print("="*60)
