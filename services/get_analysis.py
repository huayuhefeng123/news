import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import numpy as np
from .model.HybridFusionNet import HybridFusionNet
from .processer import extract_texture_features, extract_frequency_features, extract_edge_features, extract_geometric_features
from .model.GradCAM import GradCAM
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc
import cv2

def _get_analysis(image_path, output_dir="grad-cam-results-0324", image_size=224):
    # 清理 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 获取绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # # 创建输出目录
    # os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    # 构建模型路径 - 使用绝对路径
    model_path = os.path.abspath(os.path.join(current_dir, 'checkpoints', 'best_model.pth'))

    # 加载模型
    try:
        model = HybridFusionNet().to(device)
        # 使用 weights_only=True 来避免安全警告
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {'error': str(e)}
    
    # 处理图像
    print(f"Processing image: {image_path}")
    try:
        original_img = Image.open(image_path).convert('RGB')
        original_size = original_img.size
        
        # 调整图像大小用于模型输入 - 使用更小的尺寸减少内存使用
        img = original_img.resize((image_size, image_size))
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        
        # 使用 CPU 进行部分处理来减轻 GPU 负担
        img_np = img_tensor.cpu().numpy()
    except Exception as e:
        print(f"Error processing image: {e}")
        return {'error': str(e)}

    # 提取特征 - 使用更小的批量和更高效的处理
    try:
        print("Extracting features...")
        # 将图像转换为 uint8 以避免纹理特征警告
        img_np_uint8 = (img_np * 255).astype(np.uint8)
        
        # 分块处理特征提取
        with torch.no_grad():  # 确保不跟踪梯度以节省内存
            texture = extract_texture_features(img_np_uint8, batch_size=1).to(device)
            
            # 清理内存
            torch.cuda.empty_cache()
            
            frequency = extract_frequency_features(img_np).to(device)
            
            # 清理内存
            torch.cuda.empty_cache()
            
            edge = extract_edge_features(img_np).to(device)
            
            # 清理内存
            torch.cuda.empty_cache()
            
            gray_img = np.mean(img_np.squeeze(0), axis=0)
            geometric = extract_geometric_features(gray_img[None, ...]).to(device)

        inputs = {
            "image": img_tensor,
            "texture": texture,
            "frequency": frequency,
            "edge": edge,
            "geometric": geometric,
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return {'error': str(e)}

    # 获取模型预测 - 使用混合精度以减少内存使用
    try:
        print("Running prediction...")
        # 使用混合精度
        with torch.amp.autocast(device_type=device, enabled=True):
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs, dim=1)
                fake_prob = probabilities[0, 0].item()
                is_fake = fake_prob > 0.5
        
        # 清理不需要的变量以释放内存
        del probabilities
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': str(e)}
    
    # 获取模型预测和Grad-CAM时优化内存使用
    try:
        print("Generating Grad-CAM visualization...")
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        
        # 限制图像大小
        max_size = 512  # 减小最大尺寸以节省内存
        original_img_np = np.array(original_img)
        h, w = original_img_np.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            original_img = original_img.resize((new_w, new_h))
        
        # 获取目标层
        target_layer = None
        for name, module in model.named_modules():
            if name == 'backbone_convnext.stages.3':
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError("Target layer not found")
        
        # 创建GradCAM实例并生成热力图
        gradcam = GradCAM(model, target_layer)
        
        # 使用较小的批处理来减少内存使用
        with torch.cuda.amp.autocast():
            cam = gradcam(inputs)
        
        # 清理内存
        torch.cuda.empty_cache()
        
        # 调整热力图大小到原始图像尺寸
        cam = cv2.resize(cam, (original_img.size[0], original_img.size[1]))
        
        # 生成可视化
        cam_image = gradcam.visualize(original_img, cam)
        
        # 清理内存
        del gradcam, cam
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return {
            'prediction': 'Fake' if is_fake else 'Real',
            'probability': fake_prob,
            'error_gradcam': str(e)
        }
    
    # 保存结果
    try:
        # 创建完整的输出目录路径
        print(1)
        output_dir_path = os.path.abspath(os.path.join(current_dir, output_dir))
        print(2)
        os.makedirs(output_dir_path, exist_ok=True)
        # print(3)
        # print(output_dir_path)
        # print(image_path)
        # print(os.path.basename(image_path))
        # print(f"{os.path.splitext(os.path.basename(image_path))[0]}_gradcam.png")

        
        # # 构建输出文件路径
        # output_path = os.path.join(output_dir_path, 
        #                           f"{os.path.splitext(os.path.basename(image_path))[0]}_gradcam.png")
        # 方案2：直接从 FileStorage 对象中提取文件名
        from werkzeug.utils import secure_filename
        if hasattr(image_path, 'filename'):  # 检查是否为 FileStorage 对象
            # 直接使用 FileStorage 的文件名属性
            filename = secure_filename(image_path.filename)
            base_name = os.path.splitext(filename)[0]
        else:
            # 如果已经是字符串路径
            base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 构建输出文件路径
        output_path = os.path.join(output_dir_path, f"{base_name}_gradcam.png")
        print("------------------------------------------------------------")
        print(output_path)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cam_image)
        plt.axis('off')
        plt.title(f"Prediction: {'Fake' if is_fake else 'Real'} (Probability: {fake_prob:.4f})")
        
        # 添加颜色条
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cax)
        cbar.set_label('Attention Level', rotation=270, labelpad=15)
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Results saved to {output_path}")
        print(f"Prediction: {'Fake' if is_fake else 'Real'} with probability {fake_prob:.4f}")
    except Exception as e:
        print(f"Error saving results: {e}")
        output_path = None
    
    # 清理所有变量以释放内存
    del model, img_tensor, texture, frequency, edge, geometric, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    result = {
        'prediction': 'Fake' if is_fake else 'Real',
        'confidence': fake_prob,
        'output_path': output_path
    }
    
    return result

if __name__ == "__main__":
    try:
        # 必须使用模型预期的图像尺寸
        result = _get_analysis(
            image_path=r'D:\news\Fronted\news-verification-project\backend\app\services\image\dataset\0.jpg',
            image_size=224  # 使用模型预期的图像尺寸
        )
        print(result)
    except Exception as e:
        print(f"Overall error: {e}")