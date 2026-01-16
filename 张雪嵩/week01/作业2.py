import jieba #中文分词
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer #词频统计
from sklearn.neighbors import KNeighborsClassifier
#读取数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
#提取文本特征
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
#构建模型
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

#使用机器学习训练模型预测用户输入的文本
def test_classify_by_sklearn(text: str) -> str:
    """
    使用机器学习训练模型预测用户输入的文本
    :param text:
    :return: text:
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)
def test_classify_by_qwllm(text: str) -> str:
    """
    使用大模型进行文本分类
    :param text:
    :return:
    """
    client = OpenAI(
        api_key="sk-6a38ccb6fcxxxx0e6324ecaf2", # 账号绑定的
        # 大模型厂商的地址
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus", # 模型的代号
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类: {text}
                输出的类型只能从下面列表中选择：
                FilmTele-Play
                Video-Play
                Music-Play
                Radio-Listen
                Alarm-Update
                Travel-Query
                HomeAppliance-Control
                Weather-Query
                Calendar-Query
                TVProgram-Play
                Audio-Play
                Other"""
             },  # 用户的提问
            ]
        )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print(test_classify_by_sklearn("我要去北京天安门"))
    print(test_classify_by_qwllm("我要和小明去环球影城玩"))
