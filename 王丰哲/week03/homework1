import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 数据读取与预处理

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {"<pad>": 0}
for text in texts:
    for ch in text:
        if ch not in char_to_index:
            char_to_index[ch] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40

# 2. 自定义 Dataset

class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(ch, 0) for ch in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 通用 RNN / LSTM / GRU 分类模型

class RecurrentClassifier(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported rnn_type")

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, x):
        x = self.embedding(x)
        out = self.rnn(x)

        if self.rnn_type == "LSTM":
            _, (hidden, _) = out
        else:
            _, hidden = out

        logits = self.fc(hidden.squeeze(0))
        return logits

# 4. 训练 + 评估函数

def train_and_evaluate(model, dataloader, epochs=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    return acc

# 5. 三种模型自动对比实验

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
epochs = 4

results = {}

for rnn_type in ["RNN", "LSTM", "GRU"]:
    print("\n" + "=" * 60)
    print(f"开始训练模型: {rnn_type}")
    print("=" * 60)

    model = RecurrentClassifier(
        rnn_type,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim
    )

    acc = train_and_evaluate(model, dataloader, epochs)
    results[rnn_type] = acc

# 6. 显眼的最终对比输出

print("RNN / LSTM / GRU 文本分类性能对比（Accuracy）")

for name, acc in results.items():
    print(f"{name:<6} Accuracy: {acc:.4f}")

best_model = max(results, key=results.get)

print("\n" + "-" * 20)
print(f"最优模型：{best_model} （Accuracy = {results[best_model]:.4f}）")
print("-" * 20)
