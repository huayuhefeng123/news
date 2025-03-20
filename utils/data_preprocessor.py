import numpy as np
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    """
    预处理文本数据
    """
    # 创建分词器
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text])
    
    # 转换文本为序列
    sequences = tokenizer.texts_to_sequences([text])
    
    # 填充序列
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    return padded_sequences[0]

def preprocess_image(image):
    """
    预处理图像数据
    """
    # 调整图像大小
    resized_image = cv2.resize(image, (224, 224))
    
    # 归一化
    normalized_image = resized_image / 255.0
    
    return normalized_image

def preprocess_video(cap):
    """
    预处理视频数据
    """
    frames = []
    max_frames = 30
    
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 调整帧大小
        resized_frame = cv2.resize(frame, (224, 224))
        
        # 归一化
        normalized_frame = resized_frame / 255.0
        
        frames.append(normalized_frame)
    
    # 填充或截断帧序列
    if len(frames) < max_frames:
        padding = [np.zeros((224, 224, 3)) for _ in range(max_frames - len(frames))]
        frames.extend(padding)
    elif len(frames) > max_frames:
        frames = frames[:max_frames]
    
    return np.array(frames) 