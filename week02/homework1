from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

#加载数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(10))
texts = dataset[0].tolist()
text_labels = dataset[1].tolist()

#给标签编码
label_to_index = {label: i for i, label in enumerate[Any](set[Any](text_labels))}
# print(label_to_index)
'''
1.set(text_labels)

2.menumerate 产生映射

3.{label: i} 生成字典

'''
#将编码替换数据集中的标签(给数据集标签编码)
numerical_text_labels = [label_to_index[label] for label in text_labels]

#字符串编码  字符 -> index
char_to_index = {'<occupy>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

#字符串编码反转 index -> 字符
index_to_char = {i: char for char, i in char_to_index.items()}
# print(char_to_index.items())
vocab_size = len(char_to_index)
# print(vocab_size)

#保证输入神经网络长度一致，多截少补。
max_len = 40


#构建词袋
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()
        print(type(self.labels))
        print(type(self.bow_vectors))

    def _create_bow_vectors(self):
        tk_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char,0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tk_texts.append(tokenized)
        
        bow_vectors = []
        for text_indices in tk_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        '''
        词袋统计的是词频
        '''
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]



#构建神经网络
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class TowSimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TowSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class ThreeSimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ThreeSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

char_dataset = CharBoWDataset(texts, numerical_text_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

def training(hidden_dim):
    model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss() #损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx,(inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs=model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #     if idx % 50 == 0:
        #         print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        Loss = []
        Loss.append(running_loss / len(dataloader))
    return min(Loss)

def training_layer(model):

    criterion = nn.CrossEntropyLoss() #损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx,(inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs=model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #     if idx % 50 == 0:
        #         print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        Loss = []
        Loss.append(running_loss / len(dataloader))
    return min(Loss)
#loss研究：
print("不同的节点数，相同的隐藏层(1层)：")
print('-'*20)
#训练过程
hidden_dim = 128
Loss = training(hidden_dim)
print(f"{hidden_dim + output_dim + vocab_size}个节点的最佳loss为{Loss}")

hidden_dim = 256
Loss = training(hidden_dim)
print(f"{hidden_dim + output_dim + vocab_size}个节点的最佳loss为{Loss}")

hidden_dim = 512
Loss = training(hidden_dim)
print(f"{hidden_dim + output_dim + vocab_size}个节点的最佳loss为{Loss}")

hidden_dim = 1024
Loss = training(hidden_dim)
print(f"{hidden_dim + output_dim + vocab_size}个节点的最佳loss为{Loss}")
print('-'*20)

print('*'*50)

print("相同的节点数(hidden_dim=512)，不同的隐藏层：")
print('-'*20)
hidden_dim = 512
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
Loss = training_layer(model)
print(f"一层隐藏层的最佳loss为{Loss}")
model = TowSimpleClassifier(vocab_size, hidden_dim, output_dim)
Loss = training_layer(model)
print(f"二层隐藏层的最佳loss为{Loss}")
model = ThreeSimpleClassifier(vocab_size, hidden_dim, output_dim)
Loss = training_layer(model)
print(f"三层隐藏层的最佳loss为{Loss}")
print('-'*20)


