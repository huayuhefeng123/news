# 文本验证返回值示例
TEXT_VERIFICATION_EXAMPLES = [
    {
        "isReal": True,
        "confidence": 0.89,
        "type": "text",
        "content": "新华社北京3月1日电 中共中央政治局...",
        "reason": "基于语言表达特征分析、信息来源可靠性、内容逻辑性检查等多个维度的综合分析",
        "timestamp": "2024-03-01T10:30:00Z",
        "factors": {
            "语言特征": 0.92,
            "来源可靠性": 0.88,
            "内容逻辑性": 0.90,
            "关键词分析": 0.86
        },
        "analysisData": {
            "情感倾向": 0.85,
            "主题相关性": 0.92,
            "时效性": 0.88,
            "关键词权重": 0.91
        }
    },
    {
        "isReal": False,
        "confidence": 0.23,
        "type": "text",
        "content": "震惊！某知名企业发现外星人技术...",
        "reason": "基于内容真实性、信息源可靠性和语言特征分析，发现多处虚假信息特征",
        "timestamp": "2024-03-01T11:45:00Z",
        "factors": {
            "语言特征": 0.15,
            "来源可靠性": 0.20,
            "内容逻辑性": 0.25,
            "关键词分析": 0.32
        },
        "analysisData": {
            "情感倾向": 0.75,
            "主题相关性": 0.30,
            "时效性": 0.45,
            "关键词权重": 0.25
        }
    }
]

# 图片验证返回值示例
IMAGE_VERIFICATION_EXAMPLES = [
    {
        "isReal": True,
        "confidence": 0.95,
        "type": "image",
        "imageUrl": "/uploads/images/example1.jpg",
        "reason": "基于图像元数据分析、图像篡改检测、内容一致性检查等多个维度的综合分析",
        "timestamp": "2024-03-01T13:20:00Z",
        "factors": {
            "元数据完整性": 0.98,
            "篡改检测": 0.96,
            "内容一致性": 0.94,
            "图像质量": 0.92
        },
        "analysisData": {
            "EXIF数据": 0.99,
            "像素分析": 0.95,
            "压缩特征": 0.93,
            "光照一致性": 0.96
        }
    },
    {
        "isReal": False,
        "confidence": 0.15,
        "type": "image",
        "imageUrl": "/uploads/images/example2.jpg",
        "reason": "检测到明显的图像编辑痕迹，元数据异常，且存在多处不自然特征",
        "timestamp": "2024-03-01T14:10:00Z",
        "factors": {
            "元数据完整性": 0.20,
            "篡改检测": 0.10,
            "内容一致性": 0.15,
            "图像质量": 0.25
        },
        "analysisData": {
            "EXIF数据": 0.18,
            "像素分析": 0.12,
            "压缩特征": 0.22,
            "光照一致性": 0.15
        }
    }
]

# 视频验证返回值示例
VIDEO_VERIFICATION_EXAMPLES = [
    {
        "isReal": True,
        "confidence": 0.87,
        "type": "video",
        "videoUrl": "/uploads/videos/example1.mp4",
        "reason": "基于视频帧连续性分析、音视频同步性检查、编辑痕迹检测等多个维度的综合分析",
        "timestamp": "2024-03-01T15:30:00Z",
        "factors": {
            "帧连续性": 0.89,
            "音视频同步": 0.88,
            "编辑痕迹": 0.85,
            "内容真实性": 0.86
        },
        "analysisData": {
            "运动一致性": 0.90,
            "场景连贯性": 0.88,
            "音频分析": 0.86,
            "时空一致性": 0.87
        }
    },
    {
        "isReal": False,
        "confidence": 0.28,
        "type": "video",
        "videoUrl": "/uploads/videos/example2.mp4",
        "reason": "检测到明显的视频编辑痕迹，帧序列存在异常，音视频不同步",
        "timestamp": "2024-03-01T16:45:00Z",
        "factors": {
            "帧连续性": 0.25,
            "音视频同步": 0.30,
            "编辑痕迹": 0.20,
            "内容真实性": 0.35
        },
        "analysisData": {
            "运动一致性": 0.22,
            "场景连贯性": 0.28,
            "音频分析": 0.32,
            "时空一致性": 0.25
        }
    }
]

# 返回值字段说明
RESPONSE_FIELD_DESCRIPTIONS = {
    "isReal": "布尔值，表示内容是否真实",
    "confidence": "浮点数，表示置信度（0-1之间）",
    "type": "字符串，表示验证类型（text/image/video）",
    "content/imageUrl/videoUrl": "字符串，验证内容或资源URL",
    "reason": "字符串，分析结论的原因说明",
    "timestamp": "ISO格式的时间字符串",
    "factors": "对象，包含各个评估因素的得分",
    "analysisData": "对象，包含详细的分析数据"
}

# 使用示例
"""
# 文本验证接口返回示例
@verification_bp.route('/text', methods=['POST'])
def verify_text():
    # ... 验证逻辑 ...
    return jsonify(TEXT_VERIFICATION_EXAMPLES[0])  # 返回真实案例
    # 或
    return jsonify(TEXT_VERIFICATION_EXAMPLES[1])  # 返回虚假案例

# 图片验证接口返回示例
@verification_bp.route('/image', methods=['POST'])
def verify_image():
    # ... 验证逻辑 ...
    return jsonify(IMAGE_VERIFICATION_EXAMPLES[0])  # 返回真实案例
    # 或
    return jsonify(IMAGE_VERIFICATION_EXAMPLES[1])  # 返回虚假案例

# 视频验证接口返回示例
@verification_bp.route('/video', methods=['POST'])
def verify_video():
    # ... 验证逻辑 ...
    return jsonify(VIDEO_VERIFICATION_EXAMPLES[0])  # 返回真实案例
    # 或
    return jsonify(VIDEO_VERIFICATION_EXAMPLES[1])  # 返回虚假案例
""" 