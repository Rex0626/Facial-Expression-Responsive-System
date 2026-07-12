import torch

from models.resnet18_emotion import EmotionResNet18



device = "cuda" if torch.cuda.is_available() else "cpu"


model = EmotionResNet18(
    num_classes=7,
    pretrained=True
)


model.to(device)


# 測試圖片尺寸
dummy_input = torch.randn(
    4,
    1,
    224,
    224
).to(device)



output = model(dummy_input)


print("Device:")
print(device)


print("\nOutput shape:")
print(output.shape)