import cv2
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp

with open("classes.json", "r") as f:
    classes = json.load(f)

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "asl_efficientnet_b0.pth",
        map_location="cpu"
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not success:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    prediction = ""

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                x_min = min(x_min, x)
                y_min = min(y_min, y)

                x_max = max(x_max, x)
                y_max = max(y_max, y)

            padding = 30

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)

            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            hand_crop = rgb[
                y_min:y_max,
                x_min:x_max
            ]

            if hand_crop.size != 0:

                image = Image.fromarray(
                    hand_crop
                )

                image = transform(image)

                image = image.unsqueeze(0)

                with torch.no_grad():

                    output = model(image)

                    pred = output.argmax(1).item()

                prediction = classes[pred]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "ASL Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()