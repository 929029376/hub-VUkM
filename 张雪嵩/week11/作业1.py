import os
import re
import asyncio
import uuid

os.environ["OPENAI_API_KEY"] = "sk-6a38ccb6fc234175887710e6324ecaf2"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import set_default_openai_api, set_tracing_disabled, function_tool

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


@function_tool
def sentiment_analysis(text: str) -> str:
    """
    对文本进行情感分类，判断文本的情感倾向（正面、负面、中性）。
    
    Args:
        text: 需要进行情感分析的文本内容
        
    Returns:
        情感分类结果，包括情感类型和简要分析
    """
    positive_keywords = [
        '喜欢', '爱', '开心', '高兴', '快乐', '满意', '优秀', '棒', '赞',
        '完美', '精彩', '美好', '幸福', '感谢', '感激', '期待', '希望',
        '成功', '胜利', 'good', 'great', 'excellent', 'wonderful', 'happy',
        'love', 'nice', 'perfect', 'amazing', 'fantastic'
    ]
    
    negative_keywords = [
        '讨厌', '恨', '难过', '伤心', '失望', '糟糕', '差', '烂', '垃圾',
        '无聊', '烦', '愤怒', '生气', '痛苦', '悲伤', '绝望', '失败',
        '糟糕', '坏', '恶心', '厌恶', 'bad', 'terrible', 'awful', 'sad',
        'hate', 'horrible', 'poor', 'worst', 'angry', 'disappointed'
    ]
    
    text_lower = text.lower()
    
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    total_keywords = positive_count + negative_count
    
    if total_keywords == 0:
        sentiment = "中性 (Neutral)"
        confidence = "无法检测到明显情感词汇"
    elif positive_count > negative_count:
        sentiment = "正面 (Positive)"
        confidence = f"检测到 {positive_count} 个正面词汇，{negative_count} 个负面词汇"
    elif negative_count > positive_count:
        sentiment = "负面 (Negative)"
        confidence = f"检测到 {negative_count} 个负面词汇，{positive_count} 个正面词汇"
    else:
        sentiment = "中性 (Neutral)"
        confidence = f"正面和负面词汇数量相同（各 {positive_count} 个）"
    
    result = f"""情感分析结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 分析文本：{text[:100]}{'...' if len(text) > 100 else ''}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎭 情感分类：{sentiment}
📊 分析依据：{confidence}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    return result


@function_tool
def entity_recognition(text: str) -> str:
    """
    对文本进行实体识别，提取文本中的人名、地名、组织机构名等实体信息。
    
    Args:
        text: 需要进行实体识别的文本内容
        
    Returns:
        识别出的实体列表及其类型
    """
    entities = {
        "人名 (PER)": [],
        "地名 (LOC)": [],
        "组织机构 (ORG)": [],
        "时间 (TIME)": [],
        "数字 (NUM)": []
    }
    
    chinese_name_pattern = r'[\u4e00-\u9fa5]{2,4}(?=说|表示|认为|指出|介绍|称|先生|女士|教授|博士)'
    person_matches = re.findall(chinese_name_pattern, text)
    entities["人名 (PER)"].extend(person_matches)
    
    locations = [
        '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', 
        '重庆', '西安', '苏州', '天津', '长沙', '郑州', '青岛', '大连',
        '中国', '美国', '日本', '韩国', '英国', '法国', '德国', '意大利',
        'Beijing', 'Shanghai', 'New York', 'London', 'Tokyo', 'Paris'
    ]
    for loc in locations:
        if loc in text:
            entities["地名 (LOC)"].append(loc)
    
    org_pattern = r'[\u4e00-\u9fa5]{2,8}(公司|集团|银行|大学|学院|医院|政府|部门|机构|委员会|研究院|科技|科技有限)'
    org_matches = re.findall(org_pattern, text)
    entities["组织机构 (ORG)"].extend(org_matches)
    
    common_orgs = [
        '阿里巴巴', '腾讯', '百度', '华为', '小米', '字节跳动', '京东',
        '美团', '滴滴', '网易', '新浪', '搜狐', '联想', '海尔',
        'Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook', 'Tesla'
    ]
    for org in common_orgs:
        if org in text:
            entities["组织机构 (ORG)"].append(org)
    
    time_pattern = r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}日|今天|明天|昨天|上周|下周|今年|去年'
    time_matches = re.findall(time_pattern, text)
    entities["时间 (TIME)"].extend(time_matches)
    
    num_pattern = r'\d+(?:\.\d+)?(?:万|亿|百万|千万|百|千)?(?:元|美元|人|个|件|次|%)?'
    num_matches = re.findall(num_pattern, text)
    entities["数字 (NUM)"].extend([n for n in num_matches if len(n) > 1])
    
    result_lines = ["实体识别结果：", "━" * 40]
    result_lines.append(f"📝 分析文本：{text[:80]}{'...' if len(text) > 80 else ''}")
    result_lines.append("━" * 40)
    
    has_entities = False
    for entity_type, entity_list in entities.items():
        unique_entities = list(set(entity_list))
        if unique_entities:
            has_entities = True
            result_lines.append(f"\n🏷️ {entity_type}：")
            for entity in unique_entities[:5]:
                result_lines.append(f"   • {entity}")
    
    if not has_entities:
        result_lines.append("\n⚠️ 未识别到明确的实体信息")
    
    result_lines.append("\n" + "━" * 40)
    
    return "\n".join(result_lines)


sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen-max",
    instructions="""你是一个专业的情感分析专家。你的任务是分析用户输入文本的情感倾向。

当用户请求情感分析时，你需要：
1. 使用 sentiment_analysis 工具对文本进行情感分类
2. 根据工具返回的结果，向用户解释情感分析的含义
3. 如果需要，可以提供一些改进文本情感的建议

回答时请先说明你是情感分析专家，然后给出分析结果。""",
    tools=[sentiment_analysis],
)

ner_agent = Agent(
    name="ner_agent",
    model="qwen-max",
    instructions="""你是一个专业的实体识别专家。你的任务是从用户输入的文本中提取各种实体信息。

当用户请求实体识别时，你需要：
1. 使用 entity_recognition 工具对文本进行实体识别
2. 根据工具返回的结果，向用户解释识别出的实体
3. 可以补充说明这些实体之间的关系或背景信息

回答时请先说明你是实体识别专家，然后给出识别结果。""",
    tools=[entity_recognition],
)

triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="""你是一个智能路由助手，负责根据用户的请求内容，将请求转发给最合适的专家agent处理。

请根据以下规则进行路由选择：
1. 如果用户想要分析文本的情感、情绪、态度等，请转发给 sentiment_agent（情感分析专家）
2. 如果用户想要识别文本中的人名、地名、机构名等实体信息，请转发给 ner_agent（实体识别专家）
3. 如果用户的请求不明确，可以询问用户具体需要什么帮助

请根据用户的输入内容，选择合适的agent进行处理。""",
    handoffs=[sentiment_agent, ner_agent],
)


async def main():
    conversation_id = str(uuid.uuid4().hex[:16])
    
    print("=" * 60)
    print("🤖 多Agent路由系统 - 情感分类 & 实体识别")
    print("=" * 60)
    print("\n欢迎使用智能文本分析系统！我可以帮你：")
    print("  1️⃣  情感分析 - 分析文本的情感倾向（正面/负面/中性）")
    print("  2️⃣  实体识别 - 提取文本中的人名、地名、机构名等")
    print("\n请输入您想分析的文本，或输入 'quit' 退出")
    print("-" * 60)
    
    msg = input("\n👤 请输入: ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    
    while True:
        if msg.lower() in ['quit', 'exit', 'q', '退出']:
            print("\n👋 感谢使用，再见！")
            break
            
        print("\n🤖 助手: ", end="", flush=True)
        
        with trace("Multi-Agent Routing", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")
        
        inputs = result.to_input_list()
        print("-" * 60)
        
        msg = input("\n👤 请输入: ")
        if msg.lower() in ['quit', 'exit', 'q', '退出']:
            print("\n👋 感谢使用，再见！")
            break
        inputs.append({"content": msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())
