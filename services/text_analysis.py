import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
sys.path.append(str(backend_dir))

def analyze_text(text):
    """
    分析文本的真实性
    """
    try:
        # 延迟导入
        from app.services.text.get_analysis import _get_analysis
        from app.services.text.get_topk import _get_topk
        from app.services.check import assess_text_credibility
        result ={}
        result["confidence"] = _get_analysis(text)["confidence"]
        result["top_k"] = _get_topk(text)
        # result["model"] = 'wz'
        # result["type"] = "text"
        # result['model'] = assess_text_credibility(text,0.8)
        # print(result)
        # print('-------------------------')
        # print(result)
        return result
    
    except Exception as e:
        print(f"Error in analyze_text: {e}")
        return {
            'error': str(e),
            'prediction': 'Error',
            'probability': 0.0
        }

if __name__ == "__main__":
    a=analyze_text("人工智能是一个快速发展的领域，它正在改变我们的生活方式。")
    print('-----------------------')
    print(a)
