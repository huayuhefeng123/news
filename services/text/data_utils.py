import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from transformers import BertTokenizer, AutoTokenizer
from collections import Counter
import re

class NewsDataset(Dataset):
    """处理新闻数据的Dataset类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # 处理token_type_ids（如果存在）
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        return item

def clean_text(text):
    """清理文本"""
    if isinstance(text, str):
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def load_news_data(csv_path, text_column='content', label_column='label', 
                  title_column='title', combine_title_content=True):
    """
    从CSV文件加载新闻数据
    
    Args:
        csv_path: CSV文件路径
        text_column: 内容列名
        label_column: 标签列名
        title_column: 标题列名
        combine_title_content: 是否合并标题和内容
        
    Returns:
        数据DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取数据，共{len(df)}条记录")
        
        # 检查必要的列是否存在
        required_columns = [text_column, label_column]
        if combine_title_content:
            required_columns.append(title_column)
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少以下列: {missing_columns}")
        
        # 过滤掉标签为"尚无定论"的数据
        if pd.api.types.is_string_dtype(df[label_column]):
            original_len = len(df)
            df = df[df[label_column] != "尚无定论"]
            filtered_count = original_len - len(df)
            if filtered_count > 0:
                print(f"已删除 {filtered_count} 条标签为\"尚无定论\"的记录")
        
        # 清理文本
        df[text_column] = df[text_column].apply(clean_text)
        if title_column in df.columns:
            df[title_column] = df[title_column].apply(clean_text)
        
        # 合并标题和内容（如果需要）
        if combine_title_content and title_column in df.columns:
            df['combined_text'] = df[title_column] + " " + df[text_column]
            text_column = 'combined_text'
        
        # 检查并处理空值
        if df[text_column].isna().sum() > 0:
            print(f"警告: 发现{df[text_column].isna().sum()}条内容为空的记录，将被填充为空字符串")
            df[text_column] = df[text_column].fillna("")
        
        # 标签映射
        if pd.api.types.is_string_dtype(df[label_column]):
            # 创建标签到整数的映射
            unique_labels = df[label_column].unique()
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            id_to_label = {i: label for label, i in label_to_id.items()}
            
            # 英文标签映射 (用于显示)
            english_labels = {
                "谣言": "Rumor",
                "事实": "Fact"
            }
            english_id_to_label = {i: english_labels.get(label, label) for i, label in id_to_label.items()}
            
            # 应用映射
            df['label_id'] = df[label_column].map(label_to_id)
            
            print(f"标签映射: {label_to_id}")
            
            return df, text_column, 'label_id', label_to_id, english_id_to_label
        else:
            # 标签已经是数字，确保是整数
            df['label_id'] = df[label_column].astype(int)
            unique_labels = df['label_id'].unique()
            id_to_label = {i: f"Class_{i}" for i in unique_labels}
            label_to_id = {v: k for k, v in id_to_label.items()}
            
            return df, text_column, 'label_id', label_to_id, id_to_label
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

def analyze_data(df, text_column, label_column):
    """
    分析数据集
    
    Args:
        df: 数据DataFrame
        text_column: 文本列名
        label_column: 标签列名
    """
    print("\n===== 数据集统计 =====")
    print(f"总样本数: {len(df)}")
    
    # 标签分布
    label_counts = df[label_column].value_counts()
    print("\n标签分布:")
    for label, count in label_counts.items():
        print(f"- 标签 {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # 文本长度统计
    df['text_length'] = df[text_column].apply(lambda x: len(str(x)))
    
    print("\n文本长度统计:")
    print(f"- 最短文本: {df['text_length'].min()} 字符")
    print(f"- 最长文本: {df['text_length'].max()} 字符")
    print(f"- 平均长度: {df['text_length'].mean():.2f} 字符")
    print(f"- 中位长度: {df['text_length'].median()} 字符")
    
    # 检查极端值
    short_texts = df[df['text_length'] < 10]
    if len(short_texts) > 0:
        print(f"\n警告: 发现{len(short_texts)}条过短文本 (少于10个字符)")
    
    # 特征提取
    print("\n提取文本特征...")
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    print("文本单词统计:")
    print(f"- 最少单词数: {df['word_count'].min()}")
    print(f"- 最多单词数: {df['word_count'].max()}")
    print(f"- 平均单词数: {df['word_count'].mean():.2f}")
    
    return df

def prepare_dataloaders(df, text_column, label_column, tokenizer, test_size=0.2, 
                       max_length=128, batch_size=8, use_sampler=False,
                       random_state=42):
    """
    准备数据加载器
    
    Args:
        df: 数据DataFrame
        text_column: 文本列名
        label_column: 标签列名
        tokenizer: 分词器
        test_size: 测试集比例
        max_length: 最大序列长度
        batch_size: 批次大小
        use_sampler: 是否使用加权采样
        random_state: 随机种子
        
    Returns:
        训练和测试数据加载器，以及类别权重
    """
    # 分割训练集和测试集
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=df[label_column] if len(df) > 10 else None
    )
    
    print(f"训练集: {len(train_df)}条, 测试集: {len(test_df)}条")
    
    # 准备数据
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].tolist()
    test_texts = test_df[text_column].tolist()
    test_labels = test_df[label_column].tolist()
    
    # 创建数据集
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    
    # 计算类别权重（用于处理不平衡数据）
    class_weights = None
    if use_sampler:
        class_counts = Counter(train_labels)
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size
    )
    
    class_weights_tensor = None
    if class_weights:
        # 转换为张量格式，用于损失函数
        num_classes = len(class_weights)
        class_weights_tensor = torch.zeros(num_classes)
        for label, weight in class_weights.items():
            class_weights_tensor[label] = weight
    
    return train_dataloader, test_dataloader, class_weights_tensor, train_df, test_df