import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent  # 修改这里，往上多走一级到 backend 目录
sys.path.append(str(backend_dir))

def analyze_image(file):
    """
    分析图片的真实性
    """
    try:
        # 延迟导入
        from app.services.image.get_analysis import _get_analysis
        return _get_analysis(file)
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return {
            'error': str(e),
            'prediction': 'Error',
            'probability': 0.0
        }

if __name__ == "__main__":
    result = analyze_image(r'D:\news\Fronted\news-verification-project\backend\app\services\image\dataset\0.jpg')
    print(result)