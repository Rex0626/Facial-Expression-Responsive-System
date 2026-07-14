import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import mediapipe as mp
from models.resnet18_emotion import EmotionResNet18
from utils.transforms import get_val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================
# Load Model
# ==========================
def load_model():
    model = EmotionResNet18()
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

# ==========================
# Predict Emotion
# ==========================
def predict_emotion(model, face):
    image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    transform = get_val_transform()
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)
        confidence, prediction = torch.max(prob, 1)

    emotion = emotion_names[prediction.item()]
    confidence = confidence.item()

    return emotion, confidence


# ==========================
# Main Camera
# ==========================
def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.detections:
            h, w, _ = frame.shape
            for detection in result.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                x = max(0, x)
                y = max(0, y)

                face = frame[y:y+bh, x:x+bw]

                if face.size == 0:
                    continue

                emotion, confidence = predict_emotion(model, face)
                label = (f"{emotion} "f"{confidence*100:.1f}%")

                cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
                cv2.putText( frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Emotion Recognition", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()