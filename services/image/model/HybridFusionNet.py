import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
from timm import create_model
from safetensors.torch import load_file


class HybridFusionNet(nn.Module):
    def __init__(self):
        super(HybridFusionNet, self).__init__()

        # 获取绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 主干网络：EfficientNet、Swin Transformer 和 ConvNeXt Base
        self.backbone_efficientnet = create_model('tf_efficientnetv2_b2', pretrained=False, num_classes=0)       
        self.backbone_swin = create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=0)
        self.backbone_convnext = create_model('convnext_base', pretrained=False, num_classes=0)

        # 加载EfficientNetV2 B2的safetensors权重
        efficientnet_weights = load_file((os.path.join(current_dir, 'pretrain', 'tf_efficientnetv2_b2.safetensors')))
        self.backbone_efficientnet.load_state_dict(efficientnet_weights, strict=False)

        # 加载Swin Transformer的safetensors权重
        swin_weights = load_file(os.path.join(current_dir, 'pretrain', 'swin_small_patch4_window7_224.safetensors'))
        self.backbone_swin.load_state_dict(swin_weights, strict=False)

        # 加载ConvNeXt Base的safetensors权重
        convnext_weights = load_file(os.path.join(current_dir, 'pretrain', 'convnext_base.safetensors'))
        self.backbone_convnext.load_state_dict(convnext_weights, strict=False)

        # 冻结 EfficientNet 部分层
        for param in self.backbone_efficientnet.conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone_efficientnet.bn1.parameters():
            param.requires_grad = False

        # 冻结 Swin Transformer 的前两个阶段
        for param in self.backbone_swin.layers[0].parameters():
            param.requires_grad = False
        for param in self.backbone_swin.layers[1].parameters():
            param.requires_grad = False

        # 冻结 ConvNeXt 的前两个阶段
        for param in self.backbone_convnext.stem.parameters():
            param.requires_grad = False
        for param in self.backbone_convnext.stages[0].parameters():
            param.requires_grad = False

        # 调整主干输出尺寸，现在包含三个网络的特征
        convnext_feature_size = 1024  # ConvNeXt Base 的特征维度
        image_feature_size = self.backbone_efficientnet.num_features + self.backbone_swin.num_features + convnext_feature_size
        self.fc_backbone = nn.Linear(image_feature_size, 2048)

        # 纹理特征提取模块（GLCM + CNN）
        self.texture_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 32)
        )

        # 频域特征提取模块（FFT + 小波变换）
        self.frequency_fc = nn.Sequential(
            nn.Linear(128 * 128, 128),
            nn.ReLU()
        )

        # 边缘特征提取模块（HED + CNN）
        self.edge_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 64)
        )

        # 局部几何特征提取模块（LBP）
        self.geometric_fc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU()
        )

        # 自适应注意力模块
        self.attention_fc = nn.Sequential(
            nn.Linear(32 + 128 + 64 + 32, 4),
            nn.Softmax(dim=1)
        )

        # 动态融合模块
        self.fusion_fc = nn.Sequential(
            nn.Linear(2048 + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, **inputs):
        # 从输入字典中获取各个特征
        x = inputs['image']  # 主图像
        texture = inputs['texture']  # 纹理特征
        frequency = inputs['frequency']  # 频域特征
        edge = inputs['edge']  # 边缘特征
        geometric = inputs['geometric']  # 几何特征

        # 主干网络特征提取
        backbone_features_efficientnet = self.backbone_efficientnet(x)
        
        # 检查特征的形状，如果它有 4 个维度，则应用池化
        if len(backbone_features_efficientnet.shape) == 4:
            backbone_features_efficientnet = F.adaptive_avg_pool2d(backbone_features_efficientnet, (1, 1))
            backbone_features_efficientnet = backbone_features_efficientnet.view(backbone_features_efficientnet.size(0), -1)
        else:
            # 如果已经是扁平化的特征，直接使用
            backbone_features_efficientnet = backbone_features_efficientnet.view(backbone_features_efficientnet.size(0), -1)
        
        backbone_features_swin = self.backbone_swin(x)
        
        # 同样处理 swin 特征
        if len(backbone_features_swin.shape) == 4:
            backbone_features_swin = F.adaptive_avg_pool2d(backbone_features_swin, (1, 1))
            backbone_features_swin = backbone_features_swin.view(backbone_features_swin.size(0), -1)
        else:
            backbone_features_swin = backbone_features_swin.view(backbone_features_swin.size(0), -1)
            
        # 提取和处理 ConvNeXt 特征
        backbone_features_convnext = self.backbone_convnext(x)
        
        # 处理 ConvNeXt 特征
        if len(backbone_features_convnext.shape) == 4:
            backbone_features_convnext = F.adaptive_avg_pool2d(backbone_features_convnext, (1, 1))
            backbone_features_convnext = backbone_features_convnext.view(backbone_features_convnext.size(0), -1)
        else:
            backbone_features_convnext = backbone_features_convnext.view(backbone_features_convnext.size(0), -1)
    
        # 拼接主干网络的特征，现在包括三个网络
        backbone_features = torch.cat([
            backbone_features_efficientnet, 
            backbone_features_swin,
            backbone_features_convnext
        ], dim=1)
        backbone_features = self.fc_backbone(backbone_features)

        # 模态特征提取
        texture_features = self.texture_cnn(texture)
        frequency_features = self.frequency_fc(frequency.view(frequency.size(0), -1))
        edge_features = self.edge_cnn(edge)
        geometric_features = self.geometric_fc(geometric)

        # 拼接模态特征
        multimodal_features = torch.cat(
            [texture_features, frequency_features, edge_features, geometric_features], dim=1
        )

        # 自适应注意力
        attention_weights = self.attention_fc(multimodal_features)
        
        # 注意力加权
        weighted_features = torch.cat([
            texture_features * attention_weights[:, 0:1],
            frequency_features * attention_weights[:, 1:2],
            edge_features * attention_weights[:, 2:3],
            geometric_features * attention_weights[:, 3:4],
        ], dim=1)

        # 动态融合
        combined_features = torch.cat([backbone_features, weighted_features], dim=1)

        # 融合输出
        fusion_output = self.fusion_fc(combined_features)

        # 分类输出
        output = self.classifier(fusion_output)
        return output

    def get_gradcam_target_layers(self):
        """获取推荐的Grad-CAM目标层列表
        
        Returns
        -------
        dict
            包含不同特征分支的关键层及其对应的层名称
        """
        return {
            # EfficientNet backbone 分支
            'backbone_efficientnet_final': 'backbone_efficientnet.blocks.6',
            'backbone_efficientnet_mid': 'backbone_efficientnet.blocks.4',
            
            # Swin Transformer 分支
            'backbone_swin_final': 'backbone_swin.layers.3',
            'backbone_swin_mid': 'backbone_swin.layers.2',
            
            # ConvNeXt 分支
            'backbone_convnext_final': 'backbone_convnext.stages.3',
            'backbone_convnext_mid': 'backbone_convnext.stages.2',
            
            # 纹理特征分支
            'texture_conv': 'texture_cnn.0',  # 第一个卷积层
            
            # 边缘特征分支
            'edge_conv': 'edge_cnn.0',  # 第一个卷积层
            
            # 融合层
            'fusion': 'fusion_fc.0'  # 融合层的第一个线性层
        }

    def get_gradcam_layer_groups(self):
        """获取关键的Grad-CAM目标层
        
        选择最能体现模型特征关注点的层：
        backbone_efficientnet.conv_head: CNN主干网络的最后一个卷积层，体现高层语义特征
        backbone_convnext.stages.3: ConvNeXt模型的最后一个阶段，体现另一种高级特征
        
        Returns
        -------
        dict
            关键层分组
        """
        return {
            'backbone': [
                'backbone_efficientnet.conv_head',  # CNN最后的卷积层
                'backbone_convnext.stages.3'        # ConvNeXt的最后一个阶段
            ]
        }


# 测试模型
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 8
    x = torch.rand(batch_size, 3, 224, 224)  # 主图像
    texture = torch.rand(batch_size, 1, 224, 224)  # 纹理图像
    frequency = torch.rand(batch_size, 128, 128)  # 频域特征
    edge = torch.rand(batch_size, 1, 224, 224)  # 边缘图像
    geometric = torch.rand(batch_size, 256)  # 几何特征

    # 初始化模型并测试前向传播
    model = HybridFusionNet()
    output = model(image=x, texture=texture, frequency=frequency, edge=edge, geometric=geometric)
    print("模型输出尺寸:", output.shape)  # 输出尺寸应为 [batch_size, 2]