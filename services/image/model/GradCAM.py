import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None
        self.gradient = None
        self.original_size = None

        # 注册钩子
        self.hook_layers()

    def hook_layers(self):
        """在目标卷积层注册钩子，以获取激活图和梯度"""
        def forward_hook(module, input, output):
            self.feature_map = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        # 使用新的钩子注册方法
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def __del__(self):
        """清理钩子"""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, inputs):
        """计算类激活图"""
        try:
            # 确保模型处于评估模式
            self.model.eval()
            
            # 清理 GPU 内存
            torch.cuda.empty_cache()
            
            # 使用混合精度计算
            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = self.model(**inputs)
                
                # 获取预测类别
                pred_class = outputs.argmax(dim=1)
                
                # 清除之前的梯度
                self.model.zero_grad()
                
                # 计算目标类的梯度
                score = outputs[:, pred_class]
                
                # 分离计算图以节省内存
                score = score.detach().requires_grad_(True)
                
                # 反向传播
                score.backward()
                
                # 立即清理不需要的变量
                del outputs, score
                torch.cuda.empty_cache()
                
                # 生成热力图
                cam = self.generate_cam()
                
                return cam
                
        except Exception as e:
            print(f"Error in GradCAM __call__: {e}")
            # 返回空白热力图
            return np.zeros((224, 224))
        finally:
            # 确保清理内存
            torch.cuda.empty_cache()

    def generate_cam(self, original_size=None):
        """优化的 Grad-CAM 生成方法"""
        try:
            # 检查是否有有效的特征图和梯度
            if self.feature_map is None or self.gradient is None:
                print("Warning: feature_map or gradient is None")
                return np.zeros((224, 224))
            
            # 使用torch操作替代numpy操作
            with torch.no_grad():  # 避免创建不必要的计算图
                # 移动到CPU以减少GPU内存使用
                feature_map = self.feature_map.cpu()
                gradient = self.gradient.cpu()
                
                # 计算权重
                weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
                cam = F.relu(cam)  # 应用ReLU
                
                # 转换为numpy
                cam = cam.squeeze().numpy()
                
                # 释放内存
                del feature_map, gradient, weights
                torch.cuda.empty_cache()
                
                # 调整大小并归一化
                cam = cv2.resize(cam, (224, 224))
                if cam.max() - cam.min() != 0:
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                
                return cam
                
        except Exception as e:
            print(f"Error in generate_cam: {e}")
            return np.zeros((224, 224))

    def visualize(self, image, cam):
        """将 Grad-CAM 热力图与原始图像叠加，优化内存使用"""
        try:
            # 转换为较小的尺寸进行处理
            max_size = 1024  # 限制最大尺寸
            
            # 将PIL图像转换为numpy数组
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            
            # 如果图像太大，进行缩放
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_np = cv2.resize(image_np, (new_w, new_h))
                
            # 确保使用float32而不是float64
            image_np = image_np.astype(np.float32) / 255
            
            # 调整热力图大小
            cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
            cam = np.uint8(255 * cam)
            
            # 使用较小的数据类型
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            heatmap = heatmap.astype(np.float32) / 255
            
            # 释放不需要的内存
            del cam
            
            # 图像叠加
            superimposed_img = cv2.addWeighted(image_np, 0.7, heatmap, 0.3, 0)
            
            # 清理中间变量
            del heatmap, image_np
            
            return np.uint8(255 * superimposed_img)
            
        except Exception as e:
            print(f"Error in visualize: {e}")
            # 返回原始图像作为后备
            return np.array(image)
