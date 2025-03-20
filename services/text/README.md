## 离线使用/模型移植

本项目支持完全离线使用，无需从网络下载模型。您需要准备以下文件结构：

```
models/
└── bert-base-chinese/
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.txt
    ├── special_tokens_map.json (可选)
    └── model.safetensors (或 pytorch_model.bin)
```

然后使用本地路径加载模型：

```bash
python main.py --mode train \
               --data_path your_data.csv \
               --model_name ./models/bert-base-chinese \
               --offline \
               --output_dir ./output
```

`--offline` 参数会启用离线模式，确保不会尝试从网络下载模型。# 新闻真假分类系统

本项目实现了基于BERT的新闻真假分类系统，能够对新闻进行分类（谣言、事实）。

## 功能特点

- 使用预训练BERT模型进行文本分类
- 支持标题和内容的联合处理
- 处理不平衡数据集的策略
- 完整的训练、评估和推理流程
- 丰富的可视化功能
- 模型保存和加载功能
- 支持GPU和CPU训练

## 项目结构

项目分为多个模块，每个模块负责特定功能：

- `data_utils.py`: 数据加载和处理功能
- `model_utils.py`: 模型定义和相关工具
- `train_utils.py`: 训练、评估和预测功能 
- `visualization_utils.py`: 可视化工具
- `main.py`: 主程序，提供命令行接口
- `run_example.py`: 使用示例
- `requirements.txt`: 项目依赖

## 安装

1. 克隆项目
```bash
git clone <项目地址>
cd news-classification
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 准备数据

数据应为CSV格式，包含以下列：
- `news_id`: 新闻标识
- `label`: 新闻分类（谣言、事实）
- `title`: 新闻标题
- `content`: 新闻内容

注意：如果数据中包含"尚无定论"标签的记录，这些记录将被自动过滤掉。

### 训练模型

```bash
python main.py --mode train \
               --data_path your_data.csv \
               --text_column content \
               --title_column title \
               --label_column label \
               --combine_title \
               --model_name bert-base-chinese \
               --batch_size 8 \
               --learning_rate 2e-5 \
               --epochs 5 \
               --use_gpu \
               --output_dir ./output
```

### 评估模型

```bash
python main.py --mode eval \
               --data_path test_data.csv \
               --model_path ./output/best_model.pt \
               --batch_size 16 \
               --use_gpu
```

### 预测新闻

```bash
python main.py --mode predict \
               --predict_file news_to_predict.csv \
               --model_path ./output/best_model.pt \
               --batch_size 16 \
               --use_gpu
```

### 使用示例

可以直接运行示例脚本：

```bash
python run_example.py
```

## 参数说明

主要参数说明：

| 参数 | 说明 |
| ---- | ---- |
| --data_path | CSV数据文件路径 |
| --text_column | 文本内容列名 |
| --title_column | 标题列名 |
| --label_column | 标签列名 |
| --combine_title | 是否合并标题和内容 |
| --model_name | 预训练模型名称 |
| --batch_size | 批次大小 |
| --learning_rate | 学习率 |
| --epochs | 训练轮数 |
| --use_gpu | 是否使用GPU |
| --mode | 运行模式: train, eval, predict |
| --model_path | 评估或预测模式下的模型路径 |
| --predict_file | 预测模式下的输入文件路径 |
| --output_dir | 输出目录 |

更多参数可通过 `python main.py -h` 查看。

## 模型优化建议

1. **数据预处理**:
   - 清理文本中的特殊字符和HTML标签
   - 考虑使用文本增强技术扩充数据集

2. **模型调优**:
   - 调整学习率和批次大小
   - 尝试不同的BERT变种模型
   - 使用梯度累积处理更长序列

3. **处理不平衡数据**:
   - 使用加权采样
   - 调整损失函数的类别权重

4. **图表显示**:
   - 所有图表使用英文标签，解决中文显示问题
   - 如果需要显示中文，需确保系统安装了中文字体

## 输出示例

训练后将输出以下内容：
- 训练历史曲线图
- 混淆矩阵
- 标签分布图
- 分类报告
- 训练好的模型

## 注意事项

- 对于大型数据集，建议使用GPU进行训练
- 如果GPU内存不足，可以减小批次大小并增加梯度累积步数
- 对于很长的文本，可能需要增加最大序列长度，但会增加内存需求

## 未来改进

- 添加更多预训练模型选项
- 支持交叉验证
- 添加更多文本增强技术
- 实现模型蒸馏以提高推理速度
- 添加注意力可视化