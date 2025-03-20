import torch
import time
import numpy as np
from tqdm import tqdm
import contextlib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, roc_auc_score
)
from transformers import AdamW, get_linear_schedule_with_warmup

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    accumulation_steps=1,
    class_weights=None,
    early_stopping_patience=3,
    save_best=True,
    save_path=None,
    verbose=True
):
    """
    训练模型
    
    Args:
        model: NewsClassifier模型实例
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_ratio: 预热比例
        max_grad_norm: 最大梯度范数
        accumulation_steps: 梯度累积步数
        class_weights: 类别权重张量
        early_stopping_patience: 早停耐心值
        save_best: 是否保存最佳模型
        save_path: 保存路径
        verbose: 是否打印详细信息
        
    Returns:
        训练历史
    """
    # 优化器
    optimizer = AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 计算总训练步数
    total_steps = len(train_dataloader) * epochs // accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 损失函数
    if class_weights is not None and class_weights.device != model.device:
        class_weights = class_weights.to(model.device)
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # 清空GPU缓存
    if model.use_gpu:
        torch.cuda.empty_cache()
    
    # 初始评估
    if verbose:
        print("初始评估...")
        eval_metrics = evaluate_model(model, val_dataloader)
        print(f"初始验证集指标: 准确率={eval_metrics['accuracy']:.4f}, F1={eval_metrics['f1']:.4f}")
    
    # 用于早停的变量
    best_val_f1 = 0.0
    best_model_path = None
    early_stopping_counter = 0
    
    # 训练循环
    if verbose:
        print(f"开始训练，共{epochs}个epoch...")
    
    model.model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss = 0.0
        optimizer.zero_grad()
        
        if verbose:
            train_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            train_iter = train_dataloader
            
        for i, batch in enumerate(train_iter):
            try:
                # 使用混合精度训练（如果支持）
                with torch.cuda.amp.autocast() if model.use_gpu else contextlib.nullcontext():
                    # 准备输入
                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch['attention_mask'].to(model.device)
                    labels = batch['label'].to(model.device)
                    
                    # 准备token_type_ids（如果存在）
                    token_type_ids = batch.get('token_type_ids')
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(model.device)
                    
                    # 前向传播
                    outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None  # 不使用内部损失计算
                    )
                    
                    # 计算损失
                    logits = outputs.logits
                    loss = loss_fn(logits, labels) / accumulation_steps
                    train_loss += loss.item() * accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
                    
                    # 参数更新
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 定期清理内存
                    if model.use_gpu:
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                # 处理内存不足错误
                if 'out of memory' in str(e):
                    print(f'警告: GPU内存不足，尝试减小batch_size或使用CPU训练')
                    if model.use_gpu:
                        torch.cuda.empty_cache()
                        
                        # 转移到CPU
                        print("转移模型到CPU并继续训练")
                        model.use_gpu = False
                        model.device = torch.device('cpu')
                        model.model.to(model.device)
                        
                        # 重置优化器和调度器
                        optimizer = AdamW(
                            model.model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay
                        )
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=warmup_steps,
                            num_training_steps=total_steps
                        )
                        
                        # 转移类别权重
                        if class_weights is not None:
                            class_weights = class_weights.to(model.device)
                    else:
                        print("已经在CPU上，考虑进一步减小batch_size或模型大小")
                        break
                else:
                    raise e
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_dataloader)
        model.history['train_loss'].append(avg_train_loss)
        
        # 评估模型
        model.model.eval()
        eval_metrics = evaluate_model(model, val_dataloader)
        
        # 记录验证指标
        model.history['val_loss'].append(eval_metrics['loss'])
        model.history['val_accuracy'].append(eval_metrics['accuracy'])
        model.history['val_precision'].append(eval_metrics['precision'])
        model.history['val_recall'].append(eval_metrics['recall'])
        model.history['val_f1'].append(eval_metrics['f1'])
        
        # 打印评估结果
        if verbose:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} - 用时: {epoch_time:.2f}秒")
            print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {eval_metrics['loss']:.4f}")
            print(f"验证集指标: 准确率={eval_metrics['accuracy']:.4f}, F1={eval_metrics['f1']:.4f}")
            print(f"  精确率={eval_metrics['precision']:.4f}, 召回率={eval_metrics['recall']:.4f}")
            
            # 打印分类报告
            if 'report' in eval_metrics:
                print("\n分类报告:")
                print(eval_metrics['report'])
        
        # 保存最佳模型
        if save_best and eval_metrics['f1'] > best_val_f1:
            best_val_f1 = eval_metrics['f1']
            
            # 保存最佳模型
            if save_path:
                best_model_path = model.save_model(
                    path=save_path,
                    metadata={
                        'epoch': epoch + 1,
                        'val_metrics': eval_metrics
                    }
                )
            else:
                best_model_path = model.save_model(
                    metadata={
                        'epoch': epoch + 1,
                        'val_metrics': eval_metrics
                    }
                )
            
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # 早停
        if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
            print(f"触发早停! {early_stopping_patience}个epoch没有改善。")
            break
        
        # 恢复训练模式
        model.model.train()
    
    # 训练结束，加载最佳模型
    if save_best and best_model_path:
        print(f"训练完成，加载最佳模型: {best_model_path}")
        model.load_model(best_model_path)
    
    return model.history

def evaluate_model(model, dataloader):
    """
    评估模型
    
    Args:
        model: NewsClassifier模型实例
        dataloader: 数据加载器
        
    Returns:
        评估指标字典
    """
    model.model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    # 使用交叉熵损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            # 准备输入
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            
            # 准备token_type_ids（如果存在）
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(model.device)
            
            # 前向传播
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 生成分类报告
    class_names = None
    if model.id_to_label:
        class_names = [model.id_to_label[i] for i in sorted(model.id_to_label.keys())]
    
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        digits=4
    )
    
    # 计算ROC AUC（针对多分类使用OVR策略）
    roc_auc = None
    try:
        n_classes = len(np.unique(all_labels))
        if n_classes > 2:
            # 多分类ROC AUC
            roc_auc = roc_auc_score(
                np.eye(n_classes)[all_labels], all_probs,
                multi_class='ovr', average='weighted'
            )
        else:
            # 二分类ROC AUC
            roc_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
    except:
        pass
    
    # 平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 返回指标
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss,
        'confusion_matrix': cm,
        'report': report
    }
    
    if roc_auc is not None:
        metrics['roc_auc'] = roc_auc
    
    return metrics

def predict_text(model, text, raw_output=False):
    """
    预测单个文本
    
    Args:
        model: NewsClassifier模型实例
        text: 待预测的文本
        raw_output: 是否返回原始输出（概率分布）
        
    Returns:
        预测结果字典
    """
    model.model.eval()
    
    # 对文本进行编码
    encoding = model.tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=model.max_length,
        return_tensors='pt'
    )
    
    # 转移到设备
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    token_type_ids = encoding.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(model.device)
    
    # 预测
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # 获取预测标签
    label = prediction
    if model.id_to_label and prediction in model.id_to_label:
        label = model.id_to_label[prediction]
    
    # 构建结果
    result = {
        'prediction': label,
        'prediction_id': prediction,
        'confidence': confidence,
    }
    
    # 添加所有类别的概率
    if raw_output:
        class_probs = {}
        for i, prob in enumerate(probabilities[0].cpu().numpy()):
            class_name = model.id_to_label.get(i, f"Class_{i}") if model.id_to_label else f"Class_{i}"
            class_probs[class_name] = float(prob)
        result['probabilities'] = class_probs
    
    return result

def predict_batch(model, texts, batch_size=8, raw_output=False):
    """
    批量预测文本
    
    Args:
        model: NewsClassifier模型实例
        texts: 文本列表
        batch_size: 批次大小
        raw_output: 是否返回原始输出（概率分布）
        
    Returns:
        预测结果列表
    """
    from torch.utils.data import DataLoader
    from data_utils import NewsDataset
    
    model.model.eval()
    
    # 创建虚拟标签
    dummy_labels = [0] * len(texts)
    
    # 创建数据集和加载器
    dataset = NewsDataset(texts, dummy_labels, model.tokenizer, max_length=model.max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # 存储结果
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            # 准备输入
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(model.device)
            
            # 前向传播
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None
            )
            
            # 获取预测结果
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # 构建结果
    results = []
    for i, (pred, probs) in enumerate(zip(all_predictions, all_probabilities)):
        # 获取预测标签
        label = pred
        if model.id_to_label and pred in model.id_to_label:
            label = model.id_to_label[pred]
        
        result = {
            'text': texts[i],
            'prediction': label,
            'prediction_id': int(pred),
            'confidence': float(probs[pred])
        }
        
        # 添加所有类别的概率
        if raw_output:
            class_probs = {}
            for j, prob in enumerate(probs):
                class_name = model.id_to_label.get(j, f"Class_{j}") if model.id_to_label else f"Class_{j}"
                class_probs[class_name] = float(prob)
            result['probabilities'] = class_probs
        
        results.append(result)
    
    return results