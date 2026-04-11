import numpy as np
from PIL import Image
import requests
from sklearn.preprocessing import normalize
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import matplotlib.pyplot as plt

# ==================== 模型加载部分 ====================
print("正在从 Hugging Face Hub 加载中文 CLIP 模型...")
print("首次运行需要下载模型（约 500MB），请确保网络畅通")

# 从 Hugging Face Model Hub 直接加载模型（无需本地文件）
model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

print("模型加载成功！")

# ==================== 图像特征提取 ====================
image_path = "C:\\Users\\Administrator\\Desktop\\狗.jpg"
img = [Image.open(image_path)]  # 注意：包装成列表
input = processor(images=img, return_tensors='pt')  # 预处理图像

img_image_feat = []
with torch.no_grad():  # 关闭梯度计算（推理模式）
    image_feature = model.get_image_features(**input)  # 获取图像特征
    image_feature = image_feature.data.numpy()  # 转换为NumPy数组
    img_image_feat.append(image_feature)

img_image_feat = np.vstack(img_image_feat)  # 合并特征（此处只有1张图）
img_image_feat = normalize(img_image_feat)  # L2归一化

# ==================== 文本特征提取 ====================
img_texts_feat = []
texts = ['这是一只：小狗','这是一只：小猫','这是一只：小鸟','这是一只：鱼','这是一只：树']

inputs = processor(text=texts, return_tensors='pt', padding=True)  # 预处理文本
with torch.no_grad():
    text_features = model.get_text_features(**inputs)  # 获取文本特征
    text_features = text_features.data.numpy()
    img_texts_feat.append(text_features)

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)  # 同样进行L2归一化
print(f"文本特征维度: {img_texts_feat.shape}")

# ==================== 跨模态相似度计算与结果输出 ====================
# 计算图像特征与所有文本特征的点积（即余弦相似度）
sim_result = np.dot(img_image_feat[0], img_texts_feat.T)

# 对相似度结果排序，获取最匹配的索引
sim_idx = sim_result.argsort()[::-1][0]  # [::-1] 降序排列，[0] 取最高分

print("\n" + "="*50)
print("CLIP 跨模态匹配结果")
print("="*50)
print(f"图像: {image_path}")
print(f"最匹配类别: {texts[sim_idx]}")
print(f"相似度得分: {sim_result[sim_idx]:.4f}")
print(f"所有类别相似度:")
for i, (text, score) in enumerate(zip(texts, sim_result)):
    marker = "" if i == sim_idx else "  "
    print(f"  {marker} {text}: {score:.4f}")
print("="*50)
