import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class EmotionResNet18(nn.Module):

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
        else:
            self.model = resnet18(weights=None)


        # FER2013 是灰階圖片
        # 原本 ResNet 接 RGB
        # 修改第一層 convolution

        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )


        # 修改最後分類層

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)