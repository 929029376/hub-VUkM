import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 添加CUDA调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from config import Config
from data_loader import create_data_loader
from model import BertClassifier


def check_labels(df, column='label'):
    """检查标签数据"""
    print(f"数据标签统计:")
    print(f"  总数: {len(df)}")
    print(f"  唯一标签值: {sorted(df[column].unique())}")
    print(f"  标签范围: {df[column].min()} 到 {df[column].max()}")
    print(f"  类别数量: {df[column].nunique()}")

    # 检查是否有空值
    print(f"  空值数量: {df[column].isnull().sum()}")

    # 打印标签分布
    label_counts = df[column].value_counts().sort_index()
    print(f"  标签分布:")
    for label, count in label_counts.items():
        print(f"    标签 {label}: {count} 条 ({count / len(df) * 100:.2f}%)")

    return label_counts


def train_epoch(model, data_loader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 调试：检查批次数据
        if batch_idx == 0:
            print(f"批次 0 数据检查:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  attention_mask shape: {attention_mask.shape}")
            print(f"  labels shape: {labels.shape}")
            print(f"  labels unique values: {torch.unique(labels)}")
            print(f"  labels range: {labels.min().item()} to {labels.max().item()}")

        # 确保标签在有效范围内
        if labels.max().item() >= Config.num_classes or labels.min().item() < 0:
            print(f"错误: 标签值超出范围!")
            print(f"  labels: {labels}")
            print(f"  num_classes: {Config.num_classes}")
            raise ValueError("标签值超出模型输出范围")

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        # 检查模型输出
        if batch_idx == 0:
            print(f"  模型输出shape: {outputs.shape}")
            print(f"  模型输出范围: {outputs.min().item():.4f} to {outputs.max().item():.4f}")

        # 确保输出维度正确
        if outputs.shape[1] != Config.num_classes:
            print(f"错误: 模型输出维度不匹配!")
            print(f"  预期: {Config.num_classes}, 实际: {outputs.shape[1]}")
            raise ValueError(f"模型输出维度不匹配")

        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct_predictions += torch.sum(preds == labels)
        total_samples += len(labels)

        # 每10个批次打印一次进度
        if (batch_idx + 1) % 10 == 0:
            print(f'  批次 [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    return total_loss / len(data_loader), correct_predictions.double() / total_samples


def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 检查标签
            if labels.max().item() >= Config.num_classes or labels.min().item() < 0:
                print(f"验证集错误: 标签值超出范围!")
                continue

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels)
            total_samples += len(labels)

    return total_loss / max(len(data_loader), 1), correct_predictions.double() / max(total_samples, 1)


def train():
    # 创建保存目录
    if not os.path.exists(Config.save_dir):
        os.makedirs(Config.save_dir)

    # 加载数据
    print("加载数据...")
    df = pd.read_csv(Config.train_path)
    print(f"数据大小: {df.shape}")

    # 检查数据
    print("\n=== 数据检查 ===")
    print(f"列名: {df.columns.tolist()}")
    print(f"前5行数据:")
    print(df.head())

    # 检查标签
    label_counts = check_labels(df)

    # 确认类别数量
    actual_num_classes = df['label'].nunique()
    if actual_num_classes != Config.num_classes:
        print(f"\n警告: 配置的类别数 ({Config.num_classes}) 与实际数据类别数 ({actual_num_classes}) 不一致!")
        print(f"更新 Config.num_classes 为 {actual_num_classes}")
        Config.num_classes = actual_num_classes

    # 划分训练集和验证集
    print("\n划分训练集和验证集...")
    df_train, df_val = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    print(f"训练集大小: {len(df_train)}")
    print(f"验证集大小: {len(df_val)}")

    # 重新检查分割后的标签
    print("\n训练集标签分布:")
    check_labels(df_train)
    print("\n验证集标签分布:")
    check_labels(df_val)

    # 初始化tokenizer
    print("\n初始化tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(Config.model_path)

    # 初始化模型
    print("初始化模型...")
    model = BertClassifier(Config.num_classes)
    model = model.to(Config.device)

    # 打印模型结构
    print(f"\n模型结构:")
    print(model)
    print(f"\n总参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_data_loader = create_data_loader(
        df_train, tokenizer, Config.max_length, Config.batch_size
    )
    val_data_loader = create_data_loader(
        df_val, tokenizer, Config.max_length, Config.batch_size
    )

    print(f"训练批次数量: {len(train_data_loader)}")
    print(f"验证批次数量: {len(val_data_loader)}")

    # 检查一个批次的数据
    print("\n检查第一个训练批次...")
    for batch in train_data_loader:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  标签值: {batch['labels'].unique().tolist()}")
        break

    # 优化器和损失函数
    print("\n初始化优化器和损失函数...")
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    total_steps = len(train_data_loader) * Config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    print(f"学习率: {Config.learning_rate}")
    print(f"总训练步数: {total_steps}")
    print(f"预热步数: {int(0.1 * total_steps)}")

    best_accuracy = 0

    # 训练循环
    print(f"\n=== 开始训练，共 {Config.num_epochs} 轮 ===")
    for epoch in range(Config.num_epochs):
        print(f"\n--- 第 {epoch + 1}/{Config.num_epochs} 轮 ---")

        try:
            train_loss, train_acc = train_epoch(
                model, train_data_loader, criterion, optimizer, Config.device, scheduler
            )

            val_loss, val_acc = eval_model(
                model, val_data_loader, criterion, Config.device
            )

            print(f"训练结果:")
            print(f"  训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
            print(f"  验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc,
                    'appconfig': {
                        'num_classes': Config.num_classes,
                        'hidden_size': model.bert.config.hidden_size,
                        'model_path': Config.model_path,
                        'max_length': Config.max_length
                    }
                }, Config.best_model_path)

                # 保存tokenizer
                tokenizer.save_pretrained(Config.save_dir + 'tokenizer')

                best_accuracy = val_acc
                print(f"  保存最佳模型，准确率: {val_acc:.4f}")

        except Exception as e:
            print(f"第 {epoch + 1} 轮训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
            break

    print(f"\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_accuracy:.4f}")

    # 在CPU上测试模型加载
    print("\n测试模型加载...")
    try:
        checkpoint = torch.load(Config.best_model_path, map_location='cpu')
        test_model = BertClassifier(Config.num_classes)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")


if __name__ == '__main__':
    train()