import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 配置和工具函数
# ============================================================================

class Config:
    """配置类"""
    # BERT模型路径（修改为你的实际路径）
    BERT_PATH = r"D:\AI\bert-base-chinese"

    # 数据路径
    DATA_PATH = "data/dataset.csv"

    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def check_bert_model(cls):
        """检查BERT模型文件是否存在"""
        required_files = ['appconfig.json', 'pytorch_model.bin', 'vocab.txt']
        for file in required_files:
            file_path = os.path.join(cls.BERT_PATH, file)
            if not os.path.exists(file_path):
                print(f"警告: 缺少文件 {file_path}")
                return False
        return True


def clean_text(text):
    """简单的文本清洗"""
    if not isinstance(text, str):
        return ""
    # 去除特殊字符，保留中文、英文、数字和空格
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# 2. 加载和处理数据
# ============================================================================

def load_and_prepare_data(data_path, nrows=None):
    """加载和预处理数据"""
    print("=" * 60)
    print("加载数据...")

    # 读取数据
    df = pd.read_csv(data_path, sep='\t', header=None, nrows=nrows)
    print(f"数据形状: {df.shape}")

    # 查看数据
    print("\n前5行数据:")
    print(df.head())

    print("\n类别分布:")
    print(df[1].value_counts())

    # 简单的文本清洗
    print("\n清洗文本...")
    df['cleaned_text'] = df[0].apply(clean_text)

    # 编码标签
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df[1].values)

    num_classes = len(label_encoder.classes_)
    print(f"\n共有 {num_classes} 个类别:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i}: {cls}")

    return df['cleaned_text'].values, df['encoded_label'].values, label_encoder


# ============================================================================
# 3. 使用本地BERT模型
# ============================================================================
class SimpleTokenizer:
    """简单的分词"""
    def __init__(self, vocab_path):
        """从vocab.txt加载词汇表"""
        self.vocab = {}
        self.inv_vocab = {}

        print(f"加载词汇表: {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word = line.strip()
                self.vocab[word] = i
                self.inv_vocab[i] = word

        self.unk_token_id = self.vocab.get('[UNK]', 100)
        self.pad_token_id = self.vocab.get('[PAD]', 0)
        self.cls_token_id = self.vocab.get('[CLS]', 101)
        self.sep_token_id = self.vocab.get('[SEP]', 102)

        print(f"词汇表大小: {len(self.vocab)}")

    def tokenize(self, text):
        """简单分词：按字符分割（中文）"""
        # 中文按字符分割，英文按空格分割
        chars = []
        for char in text:
            chars.append(char)
        return chars

    def convert_tokens_to_ids(self, tokens):
        """token转id"""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # 尝试处理英文单词
                if token.isalpha() and len(token) > 1:
                    # 如果是英文单词，尝试分割
                    for ch in token:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
                        else:
                            ids.append(self.unk_token_id)
                else:
                    ids.append(self.unk_token_id)
        return ids

    def encode(self, text, max_length=128):
        """编码文本"""
        # 添加[CLS]和[SEP]
        tokens = ['[CLS]'] + self.tokenize(text)[:max_length - 2] + ['[SEP]']
        token_ids = self.convert_tokens_to_ids(tokens)

        # 填充或截断
        if len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]

        # 创建attention mask
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]

        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode(text, self.max_length)

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# 4. 简化的BERT模型（使用随机权重，避免加载大模型）
# ============================================================================

class SimpleBERTClassifier(nn.Module):
    """BERT分类（使用随机权重）"""
    def __init__(self, vocab_size=21128, hidden_size=768, num_classes=10, num_layers=2):
        super().__init__()
        print("初始化BERT分类...")
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # 简单的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, attention_mask):
        """前向传播"""
        # 嵌入层
        embeddings = self.embedding(input_ids)
        key_padding_mask = (attention_mask == 0)
        # 编码
        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=key_padding_mask
        )
        # 取[CLS] token（第一个token）
        cls_output = encoded[:, 0, :]
        # 分类
        logits = self.classifier(cls_output)
        # # 调整attention mask形状
        # attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # attn_mask = (1.0 - attn_mask) * -10000.0
        # # 编码
        # encoded = self.encoder(embeddings, attn_mask)
        # # 取[CLS] token（第一个token）
        # cls_output = encoded[:, 0, :]
        # # 分类
        # logits = self.classifier(cls_output)
        return logits


# ============================================================================
# 5. 训练和评估函数
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    """训练模型"""
    print("\n" + "=" * 60)
    print("开始训练...")
    model.to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    best_model_state = None
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if batch_idx % 20 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        train_accuracy = 100 * train_correct / train_total

        # 验证阶段
        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion)
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  训练损失: {train_loss / len(train_loader):.4f}, 准确率: {train_accuracy:.2f}%")
        print(f"  验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.2f}%")

        # 保存最好的模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  保存最佳模型 (准确率: {best_accuracy:.2f}%)")
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model
def evaluate_model(model, data_loader, criterion=None):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)

            outputs = model(input_ids, attention_mask)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader) if criterion else 0

    return accuracy, avg_loss


# ============================================================================
# 6. 预测函数
# ============================================================================

class SimpleBERTTextClassifier:
    """BERT文本分类"""
    def __init__(self, bert_path=None):
        self.bert_path = bert_path or Config.BERT_PATH
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        # 检查BERT路径
        if not Config.check_bert_model():
            print("警告: BERT模型文件不完整，将使用简化模型")
        # 设置设备
        self.device = Config.DEVICE
        print(f"使用设备: {self.device}")
    def prepare_data(self, texts, labels):
        """准备数据"""
        # 编码标签
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        print(f"共有 {num_classes} 个类别")
        for i, cls in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {cls}")
        return encoded_labels, num_classes
    def train(self, texts, labels, test_size=0.2, epochs=3):
        """训练分类器"""
        print("\n" + "=" * 60)
        print("准备训练BERT分类器")
        print("=" * 60)
        # 准备数据
        encoded_labels, num_classes = self.prepare_data(texts, labels)
        # 划分数据集
        X_train, X_val, y_train, y_val = train_test_split(
            texts, encoded_labels,
            test_size=test_size,
            random_state=Config.RANDOM_STATE,
            stratify=encoded_labels
        )
        print(f"\n训练集: {len(X_train)} 条")
        print(f"验证集: {len(X_val)} 条")
        # 加载tokenizer
        vocab_path = os.path.join(self.bert_path, 'vocab.txt')
        if os.path.exists(vocab_path):
            self.tokenizer = SimpleTokenizer(vocab_path)

        # 创建数据集
        train_dataset = TextDataset(X_train, y_train, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = TextDataset(X_val, y_val, self.tokenizer, Config.MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        # 创建模型
        vocab_size = len(self.tokenizer.vocab)
        self.model = SimpleBERTClassifier(
            vocab_size=vocab_size,
            hidden_size=768,
            num_classes=num_classes,
            num_layers=2
        )

        # 训练模型
        self.model = train_model(
            self.model, train_loader, val_loader,
            epochs=epochs,
            learning_rate=Config.LEARNING_RATE
        )

        # 最终评估
        final_accuracy, _ = evaluate_model(self.model, val_loader)
        print(f"\n最终验证准确率: {final_accuracy:.2f}%")
        self.is_trained = True
        print("\n训练完成！")

    def predict(self, text):
        """预测单个文本"""
        if not self.is_trained:
            raise ValueError("请先训练模型")
        self.model.eval()
        # 预处理文本
        cleaned_text = clean_text(text)
        # 编码文本
        encoding = self.tokenizer.encode(cleaned_text, Config.MAX_LENGTH)
        # 转换为张量
        input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(self.device)
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            predicted_idx = predicted.cpu().item()
        # 解码标签
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        return predicted_label
    def batch_predict(self, texts):
        """批量预测"""
        predictions = []
        for text in texts:
            try:
                pred = self.predict(text)
                predictions.append(pred)
            except Exception as e:
                print(f"预测失败: {e}")
                predictions.append("未知")
        return predictions

    def save_model(self, save_path="bert_classifier.pth"):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("没有训练好的模型可保存")

        save_data = {
            'model_state': self.model.state_dict(),
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder,
                                                                                     'classes_') else [],
            'label_encoder_fitted': True,
            'tokenizer_vocab': dict(self.tokenizer.vocab),
            'appconfig': {
                'bert_path': self.bert_path,
                'num_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0,
                'max_length': Config.MAX_LENGTH,
                'batch_size': Config.BATCH_SIZE
            }
        }

        torch.save(save_data, save_path, _use_new_zipfile_serialization=True)
        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path="bert_classifier.pth"):
        """加载模型"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        print(f"加载模型: {load_path}")
        save_data = torch.load(load_path, map_location=self.device,weights_only=True)

        # 恢复tokenizer
        vocab = save_data['tokenizer_vocab']
        self.tokenizer = SimpleTokenizer(None)
        self.tokenizer.vocab = vocab
        self.tokenizer.inv_vocab = {v: k for k, v in vocab.items()}

        # 恢复label encoder
        self.label_encoder = save_data['label_encoder']

        # 创建并加载模型
        num_classes = save_data['appconfig']['num_classes']
        vocab_size = len(vocab)

        self.model = SimpleBERTClassifier(
            vocab_size=vocab_size,
            hidden_size=768,
            num_classes=num_classes,
            num_layers=2
        )
        self.model.load_state_dict(save_data['model_state'])
        self.model.to(self.device)

        self.is_trained = True
        print(" 模型加载完成")

# ============================================================================
# 7. 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 60)
    print("BERT文本分类")
    print("=" * 60)

    # 1. 检查BERT模型
    if Config.check_bert_model():
        print("BERT模型文件完整")
    else:
        print("BERT模型文件不完整，将使用简化模型")

    # 2. 加载数据
    try:
        texts, labels, label_encoder = load_and_prepare_data(
            Config.DATA_PATH,
            nrows=2000  # 先使用少量数据测试
        )
        print(f"\n 数据加载成功，共 {len(texts)} 条数据")
    except Exception as e:
        print(f"\n 数据加载失败: {e}")
        print("请确保 data/dataset.csv 文件存在")
        return
    # 3. 创建并训练分类器
    classifier = SimpleBERTTextClassifier()
    # 训练模型
    try:
        classifier.train(
            texts=texts,
            labels=labels,
            test_size=Config.TEST_SIZE,
            epochs=Config.EPOCHS
        )
    except Exception as e:
        print(f"训练失败: {e}")
        return

    # 4. 保存模型
    classifier.save_model("simple_bert_classifier.pth")
    # 5. 测试预测
    print("\n" + "=" * 60)
    print("测试预测")
    print("=" * 60)
    test_texts = [
        "帮我导航到南山路",
        "今天的股票市场怎么样",
        "人工智能技术发展迅速",
        "篮球比赛的精彩瞬间",
        "电影院的排片时间"
    ]
    # 获取标签映射
    label_mapping = {}
    if hasattr(classifier, 'label_encoder') and classifier.label_encoder is not None:
        for i, label in enumerate(classifier.label_encoder.classes_):
            label_mapping[i] = label
    # elif saved_label_encoder is not None:
    #     for i, label in enumerate(saved_label_encoder.classes_):
    #         label_mapping[i] = label
    for i, text in enumerate(test_texts, 1):
        try:
            prediction = classifier.predict(text)
            prediction_label = label_mapping.get(prediction, f"类别_{prediction}")
            print(f"测试 {i}: '{text[:20]}...' → {prediction_label}")
        except Exception as e:
            print(f"测试 {i} 失败: {e}")

    # 6. 演示模型加载
    print("\n" + "=" * 60)
    print("演示模型加载")
    print("=" * 60)

    # 创建新的分类器实例
    new_classifier = SimpleBERTTextClassifier()

    try:
        # 加载保存的模型
        new_classifier.load_model("simple_bert_classifier.pth")

        # 测试加载的模型
        test_text = "测试文本分类"
        prediction = new_classifier.predict(test_text)
        print(f"加载模型预测: '{test_text}' → {prediction}")

        print("\n 模型保存和加载功能正常")
    except Exception as e:
        print(f"模型加载失败: {e}")

    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 检查是否有数据文件
    if os.path.exists(Config.DATA_PATH):
        # 运行完整程序
        main()
        #后续待处理。。。。。。。。。。。。。。。pass