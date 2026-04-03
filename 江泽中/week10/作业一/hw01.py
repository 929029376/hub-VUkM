import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 配置路径和参数
MODEL_PATH = "xxxx"
IMAGE_PATH = "xxx"
CANDIDATES = ["小狗", "小猫", "汽车", "树木", "花朵", "小鸟", "人类", "水果", "桌子", "椅子"]

# 加载模型
model = ChineseCLIPModel.from_pretrained(MODEL_PATH)
processor = ChineseCLIPProcessor.from_pretrained(MODEL_PATH)

# 加载并预处理图片
image = Image.open(IMAGE_PATH).convert("RGB")

# 推理
inputs = processor(text=CANDIDATES, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    probs = model(**inputs).logits_per_image.softmax(dim=1)[0]

# 输出结果
for label, prob in zip(CANDIDATES, probs):
    print(f"{label:6s}: {prob:.2%}")

best_idx = probs.argmax().item()
print(f"\n✅ 识别结果：{CANDIDATES[best_idx]} ({probs[best_idx]:.2%})")