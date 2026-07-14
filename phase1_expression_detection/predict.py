import torch
from PIL import Image
import torch.nn.functional as F
from models.resnet18_emotion import EmotionResNet18
from utils.transforms import get_val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available()else "cpu")

# Emotion Labels
emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

def load_model():
    model = EmotionResNet18()
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def predict(image_path):
    model = load_model()

    # ======================
    # Load Image
    # ======================
    image = Image.open(image_path).convert("L")
    transform = get_val_transform()
    image = transform(image)

    # 增加 batch dimension
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    # ======================
    # Prediction
    # ======================
    with torch.no_grad():
        output = model(image)
        probability = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probability, 1)

    emotion = emotion_names[prediction.item()]
    confidence = confidence.item()

    print("======================")
    print(f"Emotion: {emotion}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("======================")

if __name__ == "__main__":
    predict("test_images/sample.jpg")