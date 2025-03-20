import os
import gc
from pathlib import Path

def _get_analysis(content):
    """
    分析文本内容的真实性
    """
    try:
        # 清理内存
        gc.collect()
        
        # 延迟导入 torch 相关模块
        import torch
        import torch.nn.functional as F
        from app.services.text.model_utils import NewsClassifier
        from app.services.text.train_utils import predict_text
        
        # 设置离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # 获取绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建模型路径
        model_dir = os.path.abspath(os.path.join(current_dir, 'pretrained_models', 'bert-base-chinese'))
        model_file = os.path.abspath(os.path.join(current_dir, 'models', 'best_model.pt'))
        
        # 使用 CPU 进行预测以减少内存使用
        device = torch.device('cpu')
        
        # 加载模型
        print(model_file)
        checkpoint = torch.load(model_file, map_location=device)
        num_labels = checkpoint.get('num_labels', 2)
        
        # 初始化分类器
        classifier = NewsClassifier(
            model_name=model_dir,
            num_labels=num_labels,
            use_gpu=False  # 使用 CPU
        )
        
        # 预测
        result = predict_text(classifier, content)
        
        # 清理内存
        del classifier, checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Error in _get_analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'prediction': 'Error',
            'probability': 0.0
        }
