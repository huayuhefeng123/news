from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from torchvision.transforms import ToTensor
import glob
from model.HybridFusionNet import HybridFusionNet
from processer import *
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from model.GradCAM import *
import torch.nn.functional as F
from tqdm import tqdm

# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, feature_extraction_funcs, image_size=(224, 224)):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extraction_funcs = feature_extraction_funcs
        self.image_size = image_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB').resize(self.image_size)
        img_tensor = ToTensor()(img)

        # 将 tensor 转换为 numpy 数组并调整维度
        img_np = img_tensor.numpy()
        
        # 修改特征提取的调用方式
        texture = extract_texture_features(img_np[None, ...], batch_size=1).squeeze(0)
        frequency = extract_frequency_features(img_np[None, ...]).squeeze(0)
        edge = extract_edge_features(img_np[None, ...]).squeeze(0)
        # 对于几何特征，我们需要先转换为灰度图
        gray_img = np.mean(img_np, axis=0)  # 转换为灰度图
        geometric = extract_geometric_features(gray_img[None, ...]).squeeze(0)

        return {
            "image": img_tensor,
            "texture": texture,
            "frequency": frequency,
            "edge": edge,
            "geometric": geometric
        }, torch.tensor(label, dtype=torch.long)


def load_data(dataset_path):
    """加载数据并根据文件夹名称分配标签"""
    file_paths = []
    labels = []

    for label_name, label_value in [("real", 1), ("fake", 0)]:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder):
            raise ValueError(f"Folder '{folder}' does not exist in '{dataset_path}'.")

        image_files = glob.glob(os.path.join(folder, "*.*"))
        file_paths.extend(image_files)
        labels.extend([label_value] * len(image_files))

    return file_paths, labels


def train_or_test(mode, dataset_path, model_path=None, save_model_path="checkpoints", 
                  output_csv="results.csv", test_input=None, epochs=10, batch_size=8, lr=1e-3):
    print("Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridFusionNet().to(device)

    if mode == "test":
        if model_path is None:
            raise ValueError("In test mode, --model_path is required.")

        model.load_state_dict(torch.load(model_path))
        model.eval()

        if os.path.isfile(test_input):
            print(f"Processing single image: {test_input}")
            process_single_image(model, test_input, device)
        elif os.path.isdir(test_input):
            print(f"Processing folder: {test_input}")
            process_folder(model, test_input, output_csv, device)
        else:
            raise ValueError(f"Invalid test_input: {test_input}. Must be a file or directory.")
        return

    # Training setup
    file_paths, labels = load_data(dataset_path)
    train_files, val_files, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_files, train_labels, feature_extraction_funcs=[])
    val_dataset = CustomDataset(val_files, val_labels, feature_extraction_funcs=[])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard writer
    writer = SummaryWriter(log_dir="logs")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        # 添加进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in train_pbar:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            batch_acc = (preds == labels).float().mean().item()
            train_acc += batch_acc

            # 更新进度条信息
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.4f}'
            })

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # 验证集评估
        model.eval()
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_pbar:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                batch_acc = (preds == labels).float().mean().item()
                val_acc += batch_acc

                # 更新验证进度条信息
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # 记录到 TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存模型
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_file_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_file_path)
        print(f"Model saved to {model_file_path}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model saved with Val Acc: {best_acc:.4f}")

    writer.close()


def process_single_image(model, image_path, device, layer_groups=None):
    """处理单张图像并生成Grad-CAM热力图
    
    Parameters
    ----------
    model : HybridFusionNet
        模型实例
    image_path : str
        输入图像路径
    device : torch.device
        计算设备
    layer_groups : list of str, optional
        要可视化的层组名称列表，默认使用所有组
    """
    # 创建输出目录
    os.makedirs('grad-cam-results', exist_ok=True)
    
    # 处理图像
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size
    
    # 调整图像大小用于模型输入
    img = original_img.resize((224, 224))
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    img_np = img_tensor.cpu().numpy()

    # 修改特征提取的调用方式
    texture = extract_texture_features(img_np, batch_size=1).to(device)
    frequency = extract_frequency_features(img_np).to(device)
    edge = extract_edge_features(img_np).to(device)
    gray_img = np.mean(img_np.squeeze(0), axis=0)
    geometric = extract_geometric_features(gray_img[None, ...]).to(device)

    inputs = {
        "image": img_tensor,
        "texture": texture,
        "frequency": frequency,
        "edge": edge,
        "geometric": geometric,
    }

    # 获取模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs, dim=1)
        fake_prob = probabilities[0, 0].item()
        is_fake = fake_prob > 0.5
    
    # 获取目标层 - 修改为 ConvNeXt 网络的最后一个阶段
    # 将目标层从 'backbone_efficientnet.conv_head' 修改为 'backbone_convnext.stages.3'
    target_layer = dict(model.named_modules())['backbone_convnext.stages.3']
    gradcam = GradCAM(model, target_layer)
    
    # 重新计算梯度
    outputs = model(**inputs)
    pred_class = outputs.argmax(dim=1)
    outputs[:, pred_class].backward()
    
    # 生成热力图
    cam = gradcam.generate_cam(original_size=(original_size[1], original_size[0]))
    cam_image = gradcam.visualize(original_img, cam)
    
    # 保存结果
    output_path = os.path.join('grad-cam-results', f"{os.path.splitext(os.path.basename(image_path))[0]}_gradcam.png")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cam_image)
    plt.axis('off')
    plt.title(f"Prediction: {'Fake' if is_fake else 'Real'} ({fake_prob:.4f})")
    
    # 添加颜色条
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cax)
    cbar.set_label('Attention Level', rotation=270, labelpad=15)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return {
        'prediction': 'Fake' if is_fake else 'Real',
        'probability': fake_prob,
        'output_path': output_path
    }

def process_folder(model, folder_path, output_csv, device, target_layer='backbone_convnext.stages.3'):
    files = glob.glob(os.path.join(folder_path, "*.*"))
    results = []

    # 获取目标层 - 修改为 ConvNeXt 网络的最后一个阶段
    target_layers = dict(model.named_modules())
    if target_layer not in target_layers:
        print(f"Warning: Layer {target_layer} not found. Available layers:")
        print("\n".join(target_layers.keys()))
        target_layer = 'backbone_convnext.stages.3'  # 默认使用 ConvNeXt 的最后一个阶段
    
    target_conv_layer = target_layers[target_layer]

    for file in files:
        # 读取原始图像并获取其尺寸
        original_img = Image.open(file).convert('RGB')
        original_size = original_img.size
        
        # 调整图像大小用于模型输入
        img = original_img.resize((224, 224))
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        img_np = img_tensor.cpu().numpy()

        # 修改特征提取的调用方式
        texture = extract_texture_features(img_np, batch_size=1).to(device)
        frequency = extract_frequency_features(img_np).to(device)
        edge = extract_edge_features(img_np).to(device)
        gray_img = np.mean(img_np.squeeze(0), axis=0)
        geometric = extract_geometric_features(gray_img[None, ...]).to(device)

        inputs = {
            "image": img_tensor,
            "texture": texture,
            "frequency": frequency,
            "edge": edge,
            "geometric": geometric,
        }

        model.eval()
        gradcam = GradCAM(model, target_conv_layer)

        # 分两步进行：先进行预测，再计算 Grad-CAM
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs, dim=1)
            fake_prob = probabilities[0, 0].item()
            is_fake = fake_prob > 0.5

        # 重新计算输出用于 Grad-CAM（这次不使用 no_grad）
        outputs = model(**inputs)
        pred_class = outputs.argmax(dim=1)
        outputs[:, pred_class].backward()  # 对预测类别的得分进行反向传播
        cam = gradcam.generate_cam(original_size=(original_size[1], original_size[0]))

        # 可视化 Grad-CAM，使用原始图像
        cam_image = gradcam.visualize(original_img, cam)

        # 保存结果
        plt.figure(figsize=(12, 8))  # 调整图像大小以适应更大的分辨率
        plt.imshow(cam_image)
        plt.axis('off')
        plt.title(f"File: {os.path.basename(file)}\nFake Probability: {fake_prob:.4f}, Prediction: {'Fake' if is_fake else 'Real'}")
        
        # 添加颜色程度标签
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cax)
        cbar.set_label('Attention Level', rotation=270, labelpad=15)

        # 使用原始文件名保存
        base_name, ext = os.path.splitext(os.path.basename(file))
        output_path = os.path.join('gradcam_results', f"{base_name}_gradcam{ext}")
        os.makedirs('gradcam_results', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        results.append([os.path.basename(file), fake_prob, int(is_fake)])

        # 清除梯度，为下一张图片做准备
        model.zero_grad()

    # 保存结果到 CSV 文件
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Fake Probability", "Is Fake"])
        writer.writerows(results)
    print(f"Results saved to {output_csv}")


def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
            total_acc += (preds == labels).float().mean().item()

    total_loss /= len(data_loader) if criterion else 0
    total_acc /= len(data_loader)
    return total_loss, total_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="运行模式: train 或 test")
    parser.add_argument("--dataset_path", type=str, help="数据集文件夹路径 (train 模式必需)")
    parser.add_argument("--model_path", type=str, help="加载模型路径 (test 模式必需)")
    parser.add_argument("--save_model_path", type=str, default="checkpoints/best_model.pth", help="保存模型的路径")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="测试结果 CSV 文件路径")
    parser.add_argument("--test_input", type=str, help="测试输入文件或文件夹路径 (test 模式必需)")
    parser.add_argument("--epochs", type=int, default=10, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--target_layer", type=str, default="backbone_convnext.stages.3", 
                      help="Grad-CAM 目标层，默认为 ConvNeXt 的最后一个阶段")
    args = parser.parse_args()

    train_or_test(
        mode=args.mode,
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        save_model_path=args.save_model_path,
        output_csv=args.output_csv,
        test_input=args.test_input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )