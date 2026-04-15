import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from config import Config


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        # 1：使用from_pretrained，更安全
        self.bert = BertModel.from_pretrained(Config.model_path)

        # 2：更精确控制
        # appconfig = BertConfig.from_pretrained(Config.model_path)
        # appconfig.output_hidden_states = True  # 隐藏状态
        # self.bert = BertModel.from_pretrained(Config.model_path, appconfig=appconfig)

        self.drop = nn.Dropout(p=0.3)

        # 1：使用最后一个隐藏层的第一个token（CLS）位置
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

        # 2：添加额外的全连接层
        # self.fc1 = nn.Linear(self.bert.appconfig.hidden_size, 256)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        # 1：获取所有输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True  # 确保返回字典而不是元组
        )

        # 2：如果return_dict=False或模型版本旧，使用元组处理
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # if isinstance(outputs, tuple):
        #     # outputs[0] 是最后一层隐藏状态
        #     # outputs[1] 是池化输出（如果可用）
        #     last_hidden_state = outputs[0]
        # else:
        #     last_hidden_state = outputs.last_hidden_state

        # 获取CLS token的表示（第一个token）
        # 1：使用pooler_output（如果可用）
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_output = outputs.pooler_output
        else:
            # 2：使用最后一层隐藏状态的第一个token
            last_hidden_state = outputs.last_hidden_state
            cls_output = last_hidden_state[:, 0, :]  # 取第一个token [CLS]

        output = self.drop(cls_output)
        logits = self.fc(output)

        # 使用额外的全连接层
        # output = self.drop(cls_output)
        # output = self.fc1(output)
        # output = self.relu(output)
        # output = self.drop(output)
        # logits = self.fc2(output)

        return logits

    def extract_features(self, input_ids, attention_mask):
        """提取BERT特征，用于后续分析"""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                last_hidden_state = outputs.last_hidden_state
                features = last_hidden_state[:, 0, :]

            return features.cpu().numpy()