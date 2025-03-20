from flask import Blueprint, request, jsonify
import time
from app.test_return import TEXT_VERIFICATION_EXAMPLES, IMAGE_VERIFICATION_EXAMPLES, VIDEO_VERIFICATION_EXAMPLES

# 修改 url_prefix
verification_bp = Blueprint('verify', __name__)

@verification_bp.route('/text', methods=['POST', 'OPTIONS'])
def verify_text():
    if request.method == 'OPTIONS':
        return '', 204
        
    content = request.json.get('content')
    if not content:
        return jsonify({'error': '未提供文本内容'}), 400

    # 延迟导入
    from app.services.text_analysis import analyze_text
    try:
        result = analyze_text(content)
        return jsonify(result)
    except Exception as e:
        print(f"Error in verify_text: {e}")
        return jsonify({'error': str(e)}), 500

@verification_bp.route('/image', methods=['POST', 'OPTIONS'])
def verify_image():
    if request.method == 'OPTIONS':
        return '', 204
        
    if 'file' not in request.files:
        return jsonify({'error': '未提供图片文件'}), 400
    
    # 延迟导入
    from app.services.image_analysis import analyze_image
    try:
        file = request.files['file']
        result = analyze_image(file)
        return jsonify(result)
    except Exception as e:
        print(f"Error in verify_image: {e}")
        return jsonify({'error': str(e)}), 500

@verification_bp.route('/video', methods=['POST'])
def verify_video():
    if 'file' not in request.files:
        return jsonify({'error': '未提供视频文件'}), 400
    
    result = VIDEO_VERIFICATION_EXAMPLES[0]
    return jsonify(result) 