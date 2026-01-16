import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # 词频统计

from sklearn import neighbors
import jieba

from openai import OpenAI

# 1、读取csv文件
data = pd.read_csv("dataset.csv", sep = "\t" , names = ["test", "label"], nrows = None)


# 2、对语句进行jieba分词操作
input_sentence = data['test'].apply(lambda x: " ".join(jieba.lcut(x)))


# 3、提取文本特征
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)


#加载&训练模型
model1 = linear_model.LogisticRegression(max_iter=1000) #加载sklearn中的内置逻辑回归模型
model1.fit(input_feature, data["label"].values)

model2 = tree.DecisionTreeClassifier() #加载sklearn中的内置决策树训练模型
model2.fit(input_feature, data["label"].values)

client = OpenAI(
    api_key = "sk-831exxxxxb68a4e9547f",

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_class_machine(text: str) -> str:
    print("----------------------------------------")
    print("             机器学习预测结果              ")
    print("----------------------------------------")
    input_test = " ".join(jieba.lcut(text))
    test_feature = vector.transform([input_test])
    print("model1: LogisticRegression", model1.predict(test_feature))
    print("model2: DecisionTreeClassifier", model2.predict(test_feature))




def text_class_llm(text: str) -> str:
    print("----------------------------------------")
    print("              llm预测结果                ")
    print("----------------------------------------")

    prediction = client.chat.completions.create(
        model = "qwen-flash",

        messages=[
            {"role": "user", "content": f"""请给我进行文本分类：{text}
            最后的输出仅包含一下字段，不要有多余的其他字段，请给出最合适的类别。
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
            Other             
            """
            },
        ]
    )
    print (prediction.choices[0].message.content)

def test_lambdafunc():
    print(input_sentence)
    print(type(input_sentence))
    print(input_sentence[0][2])

def info_data():
    print(data.head(10))
    print("数据库的样本维度为：", data.shape)
    print("类别频率分布：",data["label"].value_counts())



if __name__ == '__main__':
    # info_data()
    # test_lambdafunc()
    text_class_machine('帮我播放周杰伦的音乐')
    text_class_llm('帮我播放周杰伦的音乐')








