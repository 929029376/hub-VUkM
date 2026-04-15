import requests
import base64
import json

api_key = "sk-b872dc69e65543658466dc1bcfec3233"
url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

def encode_image(image_path, max_size_mb=5):
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        data = f.read()
    if len(data) > max_size_mb * 1024 * 1024:
        print(f"警告：图片大小为 {len(data)/1024/1024:.1f} MB，可能超过限制")
    return base64.b64encode(data).decode('utf-8')

base64_image = encode_image("dog.jpg")

print("base64_image:", base64_image)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 构造 payload
payload = {
    "model": "qwen-vl-plus",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{base64_image}"},
                    {"text": "这张图片里有什么动物？请从蛇,狗、猫、鸟、马中选择一个最合适的答案。"}
                ]
            }
        ]
    },
    "parameters": {
        "result_format": "message"
    }
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print("状态码:", response.status_code)
    print("原始响应:", response.text)

    if response.status_code == 200:
        result = response.json()
        content = result['output']['choices'][0]['message']['content']
        print(f"分类结果：{content}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        try:
            error_info = response.json()
            print("错误详情:", json.dumps(error_info, ensure_ascii=False, indent=2))
        except:
            pass
except requests.exceptions.RequestException as e:
    print("网络请求异常:", e)