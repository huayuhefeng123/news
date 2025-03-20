import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import itertools
import matplotlib as mpl
import matplotlib.font_manager as fm
import platform
import os

# 解决matplotlib中文显示问题
def setup_chinese_font():
    """配置matplotlib显示中文"""
    system = platform.system()
    
    # 中文字体文件路径
    chinese_fonts = []
    
    # 为不同操作系统添加中文字体路径
    if system == 'Windows':
        chinese_fonts = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc'     # 微软雅黑
        ]
    elif system == 'Darwin':  # macOS
        chinese_fonts = [
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/PingFang.ttc'
        ]
    elif system == 'Linux':
        chinese_fonts = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
    
    # 尝试设置中文字体
    font_path = None
    for font in chinese_fonts:
        if os.path.exists(font):
            font_path = font
            break
    
    if font_path:
        # 设置中文字体
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"中文字体设置成功: {font_path}")
    else:
        print("警告: 未找到系统中文字体，将使用英文标签")

# 尝试配置中文字体
try:
    setup_chinese_font()
except Exception as e:
    print(f"配置中文字体出错: {e}，将使用英文标签")

def plot_training_history(history, save_path=None, figsize=(12, 8)):
    """
    绘制训练过程中的指标曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        figsize: 图形大小
    """
    if not history.get('train_loss'):
        print("No training history available")
        return
    
    plt.figure(figsize=figsize)
    
    # 创建2x2网格
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 绘制损失曲线
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Training Loss', marker='o')
    if history.get('val_loss'):
        ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制准确率曲线
    ax2 = axes[0, 1]
    if history.get('val_accuracy'):
        ax2.plot(history['val_accuracy'], label='Accuracy', marker='o', color='g')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制F1曲线
    ax3 = axes[1, 0]
    if history.get('val_f1'):
        ax3.plot(history['val_f1'], label='F1 Score', marker='o', color='purple')
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制精确率和召回率曲线
    ax4 = axes[1, 1]
    if history.get('val_precision') and history.get('val_recall'):
        ax4.plot(history['val_precision'], label='Precision', marker='o', color='b')
        ax4.plot(history['val_recall'], label='Recall', marker='s', color='r')
        ax4.set_title('Validation Precision and Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', 
                         cmap=plt.cm.Blues, figsize=(10, 8), save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        classes: 类别名称列表
        normalize: 是否归一化
        title: 图表标题
        cmap: 颜色映射
        figsize: 图形大小
        save_path: 保存路径
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    
    # 标记轴刻度
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # 标记数值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()

def plot_roc_curves(y_true, y_score, n_classes, class_names=None, figsize=(10, 8), save_path=None):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签（one-hot编码）
        y_score: 预测概率
        n_classes: 类别数量
        class_names: 类别名称列表
        figsize: 图形大小
        save_path: 保存路径
    """
    if class_names is None:
        class_names = [f"类别 {i}" for i in range(n_classes)]
    
    # 计算每个类别的ROC曲线和ROC面积
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # 将y_true转换为one-hot编码（如果不是）
    if len(y_true.shape) == 1:
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_bin[i, y_true[i]] = 1
    else:
        y_true_bin = y_true
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和ROC面积
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制所有ROC曲线
    plt.figure(figsize=figsize)
    
    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'微平均 ROC (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    # 绘制每个类别的ROC曲线
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('各类别的ROC曲线')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
    
    plt.show()

def plot_precision_recall_curves(y_true, y_score, n_classes, class_names=None, figsize=(10, 8), save_path=None):
    """
    绘制精确率-召回率曲线
    
    Args:
        y_true: 真实标签（one-hot编码）
        y_score: 预测概率
        n_classes: 类别数量
        class_names: 类别名称列表
        figsize: 图形大小
        save_path: 保存路径
    """
    if class_names is None:
        class_names = [f"类别 {i}" for i in range(n_classes)]
    
    # 计算每个类别的精确率-召回率曲线
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    # 将y_true转换为one-hot编码（如果不是）
    if len(y_true.shape) == 1:
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_bin[i, y_true[i]] = 1
    else:
        y_true_bin = y_true
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision[i] = np.mean(precision[i])
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=figsize)
    
    # 绘制每个类别的精确率-召回率曲线
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('各类别的精确率-召回率曲线')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"精确率-召回率曲线已保存到: {save_path}")
    
    plt.show()

def plot_label_distribution(df, label_column, label_names=None, figsize=(10, 6), save_path=None):
    """
    绘制标签分布
    
    Args:
        df: 数据DataFrame
        label_column: 标签列名
        label_names: 标签名称字典 {标签ID: 标签名称}
        figsize: 图形大小
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    # 计算标签计数
    label_counts = df[label_column].value_counts().sort_index()
    
    # 设置标签名称
    labels = label_counts.index
    if label_names:
        labels = [label_names.get(label, label) for label in labels]
    
    # 计算百分比
    percentages = [(count / len(df)) * 100 for count in label_counts]
    
    # 绘制柱状图
    bars = plt.bar(range(len(label_counts)), label_counts, color=sns.color_palette("husl", len(label_counts)))
    
    # 添加数值标签
    for i, (count, percentage) in enumerate(zip(label_counts, percentages)):
        plt.text(i, count + 0.1, f"{count}\n({percentage:.1f}%)", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Data Label Distribution', fontsize=15)
    plt.xlabel('Label Category', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.xticks(range(len(label_counts)), labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据表格
    table_data = []
    for label, count, percentage in zip(labels, label_counts, percentages):
        table_data.append([label, count, f"{percentage:.1f}%"])
    
    plt.table(cellText=table_data,
             colLabels=['Label', 'Count', 'Percentage'],
             cellLoc='center',
             loc='bottom',
             bbox=[0.0, -0.35, 1.0, 0.2])
    
    plt.subplots_adjust(bottom=0.25)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"标签分布图已保存到: {save_path}")
    
    plt.show()

def plot_text_length_distribution(df, text_column, label_column=None, bins=20, figsize=(12, 8), save_path=None):
    """
    绘制文本长度分布
    
    Args:
        df: 数据DataFrame
        text_column: 文本列名
        label_column: 标签列名（可选，用于分组）
        bins: 直方图的箱数
        figsize: 图形大小
        save_path: 保存路径
    """
    # 计算文本长度
    df['text_length'] = df[text_column].apply(lambda x: len(str(x)))
    
    plt.figure(figsize=figsize)
    
    if label_column and label_column in df.columns:
        # 按标签分组绘制
        unique_labels = df[label_column].unique()
        
        for label in unique_labels:
            subset = df[df[label_column] == label]
            sns.histplot(subset['text_length'], bins=bins, alpha=0.5, label=f"Label {label}")
        
        plt.title(f'Text Length Distribution by Label', fontsize=15)
        plt.legend()
    else:
        # 不分组绘制
        sns.histplot(df['text_length'], bins=bins, kde=True)
        plt.title('Text Length Distribution', fontsize=15)
    
    plt.xlabel('Text Length (Characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加统计信息
    stats_text = (
        f"Min: {df['text_length'].min()}\n"
        f"Max: {df['text_length'].max()}\n"
        f"Mean: {df['text_length'].mean():.2f}\n"
        f"Median: {df['text_length'].median()}\n"
        f"Std Dev: {df['text_length'].std():.2f}"
    )
    
    plt.text(0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"文本长度分布图已保存到: {save_path}")
    
    plt.show()

def plot_word_cloud(df, text_column, max_words=100, background_color='white', 
                   figsize=(12, 8), save_path=None):
    """
    绘制词云图
    
    Args:
        df: 数据DataFrame
        text_column: 文本列名
        max_words: 词云中显示的最大单词数
        background_color: 背景颜色
        figsize: 图形大小
        save_path: 保存路径
    """
    try:
        from wordcloud import WordCloud
        import jieba
        
        # 合并所有文本
        text = ' '.join(df[text_column].dropna().astype(str))
        
        # 中文分词
        seg_list = jieba.cut(text)
        seg_text = ' '.join(seg_list)
        
        # 生成词云
        wordcloud = WordCloud(font_path='SimHei.ttf',  # 中文字体，请确保系统中有此字体
                            max_words=max_words,
                            background_color=background_color,
                            width=800, height=400,
                            random_state=42)
        wordcloud.generate(seg_text)
        
        # 显示词云
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('词云图', fontsize=15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"词云图已保存到: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("需要安装wordcloud和jieba库: pip install wordcloud jieba")