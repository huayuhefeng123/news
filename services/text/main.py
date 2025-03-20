import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 导入自定义模块
from data_utils import (
    load_news_data, analyze_data, prepare_dataloaders
)
from model_utils import NewsClassifier, set_seed
from train_utils import (
    train_model, evaluate_model, predict_text, predict_batch
)
from visualization_utils import (
    plot_training_history, plot_confusion_matrix, 
    plot_label_distribution, plot_text_length_distribution
)

def setup_arg_parser():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='新闻真假分类模型训练与评估')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, 
                       help='CSV数据文件路径')
    parser.add_argument('--text_column', type=str, default='content',
                       help='文本内容列名')
    parser.add_argument('--title_column', type=str, default='title',
                       help='标题列名')
    parser.add_argument('--label_column', type=str, default='label',
                       help='标签列名')
    parser.add_argument('--combine_title', action='store_true',
                       help='是否合并标题和内容')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大序列长度')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='./pretrained_models/bert-base-chinese',
                       help='预训练模型名称或本地路径')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    
    # 训练参数
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--use_sampler', action='store_true',
                       help='是否使用加权采样器处理不平衡数据')
    parser.add_argument('--use_gpu', action='store_true',
                       help='是否使用GPU')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--early_stopping', type=int, default=3,
                       help='早停耐心值')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict'], 
                       default='train', help='运行模式: train, eval, predict')
    parser.add_argument('--model_path', type=str, default=None,
                       help='评估或预测模式下的模型路径')
    parser.add_argument('--predict_file', type=str, default=None,
                       help='预测模式下的输入文件路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    
    # 离线模式参数
    parser.add_argument('--offline', action='store_true',
                       help='是否使用离线模式，不从网络下载模型')
    
    return parser

def train_news_classifier(args):
    """训练新闻分类器"""
    print("\n===== 开始训练模式 =====")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df, text_col, label_col, label_to_id, id_to_label = load_news_data(
        args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        title_column=args.title_column,
        combine_title_content=args.combine_title
    )
    
    # 分析数据
    df = analyze_data(df, text_col, label_col)
    
    # # 绘制标签分布图
    # plot_label_distribution(
    #     df, label_col, id_to_label,
    #     save_path=os.path.join(output_dir, "label_distribution.png")
    # )
    
    # # 绘制文本长度分布图
    # plot_text_length_distribution(
    #     df, text_col, label_col,
    #     save_path=os.path.join(output_dir, "text_length_distribution.png")
    # )
    
    # 初始化分类器
    classifier = NewsClassifier(
        model_name=args.model_name,
        num_labels=len(label_to_id),
        dropout_rate=args.dropout,
        max_length=args.max_length,
        use_gpu=args.use_gpu,
        model_dir=output_dir
    )
    
    # 设置标签映射
    classifier.set_label_mapping(label_to_id, id_to_label)
    
    # 准备数据加载器
    train_dataloader, test_dataloader, class_weights, train_df, test_df = prepare_dataloaders(
        df, text_col, label_col, classifier.get_tokenizer(),
        test_size=args.test_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_sampler=args.use_sampler
    )
    
    # 训练模型
    history = train_model(
        classifier,
        train_dataloader,
        test_dataloader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        class_weights=class_weights,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        save_path=os.path.join(output_dir, f"best_model.pt"),
        verbose=True
    )
    
    # 绘制训练历史图
    plot_training_history(
        history,
        save_path=os.path.join(output_dir, "training_history.png")
    )
    
    # 最终评估
    print("\n===== 最终模型评估 =====")
    final_metrics = evaluate_model(classifier, test_dataloader)
    
    # 绘制混淆矩阵
    if 'confusion_matrix' in final_metrics:
        class_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
        plot_confusion_matrix(
            final_metrics['confusion_matrix'],
            classes=class_names,
            normalize=True,
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )
    
    # 保存评估结果
    with open(os.path.join(output_dir, "evaluation_results.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {final_metrics['f1']:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"Recall: {final_metrics['recall']:.4f}\n")
        if 'roc_auc' in final_metrics:
            f.write(f"ROC AUC: {final_metrics['roc_auc']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(final_metrics['report'])
    
    print(f"训练和评估完成! 所有输出保存在: {output_dir}")
    return classifier, output_dir

def evaluate_news_classifier(args):
    """评估新闻分类器"""
    print("\n===== 开始评估模式 =====")
    
    if not args.model_path:
        print("错误: 评估模式需要指定模型路径 (--model_path)")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df, text_col, label_col, label_to_id, id_to_label = load_news_data(
        args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        title_column=args.title_column,
        combine_title_content=args.combine_title
    )
    
    # 初始化分类器并加载模型
    classifier = NewsClassifier(
        model_name='bert-base-chinese',  # 会被加载的模型覆盖
        use_gpu=args.use_gpu
    )
    classifier.load_model(args.model_path)
    
    # 准备数据加载器
    _, test_dataloader, _, _, test_df = prepare_dataloaders(
        df, text_col, label_col, classifier.get_tokenizer(),
        test_size=1.0,  # 评估模式使用全部数据
        max_length=classifier.max_length,
        batch_size=args.batch_size
    )
    
    # 评估模型
    metrics = evaluate_model(classifier, test_dataloader)
    
    # 绘制混淆矩阵
    if 'confusion_matrix' in metrics:
        class_names = [classifier.id_to_label[i] for i in sorted(classifier.id_to_label.keys())]
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            classes=class_names,
            normalize=True,
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )
    
    # 保存评估结果
    with open(os.path.join(output_dir, "evaluation_results.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        if 'roc_auc' in metrics:
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(metrics['report'])
    
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    print(f"评估完成! 所有输出保存在: {output_dir}")
    return classifier, output_dir

def predict_with_classifier(args):
    """使用新闻分类器进行预测"""
    print("\n===== 开始预测模式 =====")
    
    if not args.model_path:
        print("错误: 预测模式需要指定模型路径 (--model_path)")
        return
    
    if not args.predict_file:
        print("错误: 预测模式需要指定输入文件路径 (--predict_file)")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"predict_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载预测数据
    try:
        predict_df = pd.read_csv(args.predict_file)
        print(f"成功读取预测数据，共{len(predict_df)}条记录")
    except Exception as e:
        print(f"读取预测数据出错: {e}")
        return
    
    # 初始化分类器并加载模型
    classifier = NewsClassifier(
        model_name='bert-base-chinese',  # 会被加载的模型覆盖
        use_gpu=args.use_gpu
    )
    classifier.load_model(args.model_path)
    
    # 准备文本数据
    text_data = []
    if args.combine_title and args.title_column in predict_df.columns and args.text_column in predict_df.columns:
        text_data = (predict_df[args.title_column] + " " + predict_df[args.text_column]).tolist()
    elif args.text_column in predict_df.columns:
        text_data = predict_df[args.text_column].tolist()
    else:
        print(f"错误: 预测数据中缺少必要的列")
        return
    
    # 批量预测
    print(f"开始预测 {len(text_data)} 条记录...")
    predictions = predict_batch(
        classifier, 
        text_data, 
        batch_size=args.batch_size, 
        raw_output=True
    )
    
    # 创建结果DataFrame
    results_df = predict_df.copy()
    
    # 添加预测结果列
    results_df['prediction'] = [pred['prediction'] for pred in predictions]
    results_df['confidence'] = [pred['confidence'] for pred in predictions]
    
    # 添加各类别概率列
    if 'probabilities' in predictions[0]:
        for class_name in predictions[0]['probabilities'].keys():
            prob_col = f'prob_{class_name}'
            results_df[prob_col] = [pred['probabilities'][class_name] for pred in predictions]
    
    # 保存预测结果
    results_path = os.path.join(output_dir, "prediction_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # 分析预测结果
    pred_counts = results_df['prediction'].value_counts()
    print("\n===== 预测结果统计 =====")
    for label, count in pred_counts.items():
        print(f"{label}: {count} ({count/len(results_df)*100:.2f}%)")
    
    # 如果有实际标签，计算准确率
    if args.label_column in predict_df.columns:
        from sklearn.metrics import accuracy_score
        if classifier.label_to_id:
            # 将文本标签转换为ID以进行比较
            actual_labels = predict_df[args.label_column].map(classifier.label_to_id).values
            pred_labels = [pred['prediction_id'] for pred in predictions]
        else:
            # 直接比较文本标签
            actual_labels = predict_df[args.label_column].values
            pred_labels = [pred['prediction'] for pred in predictions]
        
        accuracy = accuracy_score(actual_labels, pred_labels)
        print(f"\n准确率: {accuracy:.4f}")
    
    print(f"预测完成! 结果保存在: {results_path}")
    return results_df, output_dir

def main():
    """主函数"""
    # 解析命令行参数
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(42)
    
    # 设置离线模式
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("已启用离线模式，将从本地加载模型")
    
    # 根据模式选择操作
    try:
        if args.mode == 'train':
            classifier, output_dir = train_news_classifier(args)
        elif args.mode == 'eval':
            classifier, output_dir = evaluate_news_classifier(args)
        elif args.mode == 'predict':
            results_df, output_dir = predict_with_classifier(args)
        else:
            print(f"未知模式: {args.mode}")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())