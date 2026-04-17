import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import logging
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 配置类
class Config:
    # 模型路径
    model_path = 'D:\\AI\\bert-base-chinese'

    # 训练参数
    max_length = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存路径
    save_dir = 'models/'
    best_model_path = 'models/best_model.pth'

    # 类别映射
    category_mapping = {
        0: "科技",
        1: "金融",
        2: "医疗",
        3: "教育",
        4: "旅游",
        5: "其他"
    }

    @classmethod
    def get_num_classes(cls):
        """获取类别数量"""
        try:
            if os.path.exists('datasets/train.csv'):
                import pandas as pd
                df = pd.read_csv('datasets/train.csv')
                if 'label' in df.columns:
                    return df['label'].nunique()
        except:
            pass
        return len(cls.category_mapping)


# BERT分类器模型
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(Config.model_path)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_output = outputs.pooler_output
        else:
            cls_output = outputs.last_hidden_state[:, 0, :]

        output = self.drop(cls_output)
        return self.fc(output)


class BERTTextClassifier:
    """BERT文本分类器"""

    def __init__(self, model_path: str = None):
        self.device = Config.device
        self.model_path = model_path or Config.best_model_path
        self.num_classes = Config.get_num_classes()

        logger.info(f"初始化BERT文本分类器，类别数: {self.num_classes}")

        # 加载tokenizer和模型
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        logger.info("BERT文本分类器初始化完成")

    def _load_tokenizer(self) -> BertTokenizer:
        """加载tokenizer"""
        try:
            local_tokenizer_path = os.path.join(Config.save_dir, 'tokenizer')
            if os.path.exists(local_tokenizer_path):
                return BertTokenizer.from_pretrained(local_tokenizer_path)
            return BertTokenizer.from_pretrained(Config.model_path)
        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            raise

    def _load_model(self) -> BertClassifier:
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 获取实际类别数
            if 'appconfig' in checkpoint:
                checkpoint_num_classes = checkpoint['appconfig'].get('num_classes')
                if checkpoint_num_classes is not None:
                    self.num_classes = checkpoint_num_classes

            # 创建模型并加载权重
            model = BertClassifier(self.num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        预测单个文本，只返回最高概率的类别
        Args:
            text: 输入文本
        Returns:
            字典包含预测类别和置信度
        """
        try:
            # 预处理
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=Config.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            # 预测
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_class = torch.max(outputs, dim=1)

                predicted_class = predicted_class.item()
                confidence = probabilities[0][predicted_class].item()
            # 获取类别名称
            category_name = self.get_category_name(predicted_class)
            return {
                "predicted_class": predicted_class,
                "category": category_name,
                "confidence": round(confidence, 4)
            }

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                "predicted_class": -1,
                "category": "预测失败",
                "confidence": 0.0,
                "error": str(e)
            }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            result["text"] = text  # 添加原始文本
            results.append(result)
        return results

    def get_category_name(self, class_idx: int) -> str:
        """获取类别名称"""
        if class_idx in Config.category_mapping:
            return Config.category_mapping[class_idx]
        return f"类别_{class_idx}"

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "BERT分类器",
            "num_classes": self.num_classes,
            "category_mapping": Config.category_mapping
        }


# 创建全局实例
_bert_classifier_instance = None


def get_classifier() -> BERTTextClassifier:
    """获取分类器实例"""
    global _bert_classifier_instance
    if _bert_classifier_instance is None:
        _bert_classifier_instance = BERTTextClassifier()
    return _bert_classifier_instance

# API调用函数 - 只返回最高概率的类别
def text_classify_bert(text: str) -> Dict[str, Any]:
    """
    使用BERT进行文本分类
    Args:
        text: 输入文本
    Returns:
        dict: {
            "predicted_class": int,  # 预测的类别
            "category": str,         # 类别名称
            "confidence": float      # 置信度
        }
    """
    try:
        classifier = get_classifier()
        return classifier.predict_single(text)
    except Exception as e:
        return {
            "predicted_class": -1,
            "category": "预测失败",
            "confidence": 0.0,
            "error": str(e)
        }

def text_classify_bert_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    批量文本分类
    Args:
        texts: 文本列表

    Returns:
        list: 每个文本的预测结果
    """
    try:
        classifier = get_classifier()
        return classifier.predict_batch(texts)
    except Exception as e:
        return [{
            "text": text,
            "predicted_class": -1,
            "category": "预测失败",
            "confidence": 0.0,
            "error": str(e)
        } for text in texts]


def test_classifier():
    """测试分类器"""
    logger.info("测试文本分类器...")

    test_texts = [
        "人工智能技术正在快速发展",
        "旅游市场前景广阔",
        "科技创新推动经济发展"
    ]

    for text in test_texts:
        result = text_classify_bert(text)
        print(f"文本: {text}")
        print(f"结果: {result}")
        print("-" * 50)


if __name__ == "__main__":
    test_classifier()