import os
import torch
import random
import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer
)
import json
from datetime import datetime
import contextlib
root = os.path.dirname(__file__)
print(root)
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class NewsClassifier:
    """新闻文本分类器类"""
    
    def __init__(
        self,
        model_name=os.path.join(root,'pretrained_models','bert-base-chinese'),
        num_labels=3,
        dropout_rate=0.1,
        max_length=128,
        use_gpu=True,
        model_dir='./models'
    ):
        """
        初始化分类器
        
        Args:
            model_name: 预训练模型名称
            num_labels: 标签数量
            dropout_rate: Dropout率
            max_length: 最大序列长度
            use_gpu: 是否使用GPU
            model_dir: 模型保存目录
        """
        # 设置随机种子
        set_seed()
        
        # 设置设备
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置模型参数
        self.model_name = model_name
        self.max_length = max_length
        self.model_dir = model_dir
        self.num_labels = num_labels
        # os.makedirs(model_dir, exist_ok=True)
        
        # 加载分词器和模型
        self._load_tokenizer_and_model(model_name, num_labels, dropout_rate)
        
        # 初始化训练记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        # 标签映射
        self.label_to_id = None
        self.id_to_label = None
    
    def _load_tokenizer_and_model(self, model_name, num_labels, dropout_rate):
        """
        加载分词器和模型
        
        Args:
            model_name: 模型名称或本地路径
            num_labels: 标签数量
            dropout_rate: Dropout率
        """
        try:
            print(f"加载模型: {model_name}")
            
            # 检查是否是本地路径
            is_local_path = os.path.exists(model_name)
            print(os.path.basename(model_name))
            if is_local_path:
                print(f"从本地路径加载模型: {model_name}")
            
            # 根据模型类型选择加载方法
            if 'bert' in model_name.lower():
                try:
                    self.tokenizer = BertTokenizer.from_pretrained(model_name)
                    config = BertConfig.from_pretrained(
                        model_name,
                        num_labels=num_labels,
                        hidden_dropout_prob=dropout_rate,
                        attention_probs_dropout_prob=dropout_rate
                    )
                    
                    # 检查是否有.safetensors文件
                    if is_local_path and os.path.exists(os.path.join(model_name, "model.safetensors")):
                        print("检测到safetensors格式的模型权重")
                        # 使用safetensors格式的权重
                        self.model = BertForSequenceClassification.from_pretrained(
                            model_name,
                            config=config,
                            from_tf=False
                        )
                    else:
                        self.model = BertForSequenceClassification.from_pretrained(
                            model_name,
                            config=config
                        )
                except Exception as e:
                    print(f"加载BERT模型时出错: {e}")
                    raise
            else:
                # 对于其他类型的模型，使用Auto类
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels
                )
                # 尝试设置dropout（如果模型支持）
                if hasattr(self.model.config, 'hidden_dropout_prob'):
                    self.model.config.hidden_dropout_prob = dropout_rate
                if hasattr(self.model.config, 'attention_probs_dropout_prob'):
                    self.model.config.attention_probs_dropout_prob = dropout_rate
            
            # 转移模型到设备
            self.model.to(self.device)
            
        except Exception as e:
            print(f"加载模型时出错: {e}")

    
    def set_label_mapping(self, label_to_id, id_to_label):
        """
        设置标签映射
        
        Args:
            label_to_id: 标签到ID的映射
            id_to_label: ID到标签的映射
        """
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.num_labels = len(label_to_id)
        print(f"设置标签映射: {self.label_to_id}")
    
    def save_model(self, path=None, metadata=None):
        """
        保存模型
        
        Args:
            path: 保存路径，如果为None则使用时间戳
            metadata: 额外的元数据
            
        Returns:
            保存的路径
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.model_dir, f"news_classifier_{timestamp}.pt")
        
        # 构建保存数据
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'history': self.history,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label
        }
        
        # 添加额外元数据
        if metadata:
            save_data.update(metadata)
        
        # 保存模型
        torch.save(save_data, path)
        
        # 保存标签映射为JSON（方便查看）
        if self.label_to_id:
            label_map_path = os.path.splitext(path)[0] + "_labels.json"
            with open(label_map_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'label_to_id': self.label_to_id,
                    'id_to_label': self.id_to_label
                }, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {path}")
        return path
    
    def load_model(self, path=os.path.join(root,'models','best_model.pt')):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        try:
            print(f"从{path}加载模型...")
            checkpoint = torch.load(path, map_location=self.device)
            
            # 更新模型参数
            self.model_name = checkpoint.get('model_name', self.model_name)
            self.max_length = checkpoint.get('max_length', self.max_length)
            self.num_labels = checkpoint.get('num_labels', self.num_labels)
            self.label_to_id = checkpoint.get('label_to_id', self.label_to_id)
            self.id_to_label = checkpoint.get('id_to_label', self.id_to_label)
            
            # 如果模型架构不同，需要重新加载模型
            if self.model.config.num_labels != self.num_labels:
                print(f"重新加载模型以匹配标签数量: {self.num_labels}")
                self._load_tokenizer_and_model(self.model_name, self.num_labels, 0.1)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载历史记录（如果有）
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            
            print("模型加载成功")
            
            # 打印标签映射
            if self.label_to_id:
                print(f"标签映射: {self.label_to_id}")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
    
    def get_tokenizer(self):
        """获取分词器"""
        return self.tokenizer