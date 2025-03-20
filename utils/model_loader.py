import tensorflow as tf
import os

def load_text_model():
    """
    加载文本验证模型
    """
    # 这里使用一个简单的模型进行演示
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def load_image_model():
    """
    加载图像验证模型
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def load_video_model():
    """
    加载视频验证模型
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, 3, activation='relu', input_shape=(None, 224, 224, 3)),
        tf.keras.layers.MaxPooling3D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model 