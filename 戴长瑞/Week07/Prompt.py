import openai
import json

client = openai.OpenAI(
    api_key="sk-78cc4e9ac8f44efdb207b7232ed8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = "**"
user_input = "我想听刘德华的忘情水"

response = client.chat.completions.create(
    model="qwen-plus",          # 或其他支持的模型
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户：{user_input}"}
    ],
    temperature=0.1,            # 降低随机性
    response_format={"type": "json_object"}  # 如果模型支持强制 JSON 输出
)

# 解析结果
result = json.loads(response.choices[0].message.content)
print(result)
# 预期输出：{"domain": "music", "intent": "PLAY", "slots": {"artist": "刘德华", "song": "忘情水"}}