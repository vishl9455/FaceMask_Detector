import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("model.keras")

from ultralytics import YOLO

face_model = YOLO("model.pt")


def detect_face_mask(img):

    img = img.reshape(1, 224, 224, 3)

    prediction = model.predict(img, verbose=0)

    print("Prediction:", prediction)

    if prediction[0][0] >= 0.5:
        return 1
    else:
        return 0


def draw_label(img, text, pos, bg_color):

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        2
    )[0]

    end_x = pos[0] + text_size[0] + 10
    end_y = pos[1] - text_size[1] - 10

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)

    cv2.putText(
    img,
    text,
    (pos[0] + 5, pos[1] - 5),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2,
    cv2.LINE_AA)





cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = face_model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face.astype("float32") / 255.0

            y_pred = detect_face_mask(face)

            if y_pred == 0:
                color = (0, 255, 0)
                text = "Mask"
            else:
                color = (0, 0, 255)
                text = "No Mask"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, text, (x1, y1), color)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)