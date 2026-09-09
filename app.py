from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tensorflow.keras.models import load_model
from ultralytics import YOLO

import numpy as np
import cv2


app = FastAPI()


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

mask_model = load_model("model.keras")
face_model = YOLO("model.pt")


# ==========================================
# SERVE FRONTEND
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/")
def home():
    return FileResponse("static/index.html")


# ==========================================
# MODEL STATUS
# ==========================================

@app.get("/model-status")
def model_status():

    return {
        "keras_model": "loaded",
        "yolo_model": "loaded"
    }


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------
    # 1. Read uploaded image
    # --------------------------------------

    image_bytes = await file.read()


    # --------------------------------------
    # 2. Convert image bytes to NumPy array
    # --------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # --------------------------------------
    # 3. Convert NumPy array to OpenCV image
    # --------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        return {
            "error": "Invalid image"
        }


    # --------------------------------------
    # 4. Detect faces using YOLO
    # --------------------------------------

    results = face_model(
        frame,
        verbose=False
    )


    predictions = []


    # --------------------------------------
    # 5. Process every detected face
    # --------------------------------------

    for result in results:

        for box in result.boxes:

            # Get bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Image dimensions
            h, w = frame.shape[:2]


            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)


            # ----------------------------------
            # 6. Crop face
            # ----------------------------------

            face = frame[
                y1:y2,
                x1:x2
            ]


            if face.size == 0:
                continue


            # ----------------------------------
            # 7. Preprocess face
            # ----------------------------------

            face = cv2.resize(
                face,
                (224, 224)
            )


            face = face.astype(
                "float32"
            ) / 255.0


            face = face.reshape(
                1,
                224,
                224,
                3
            )


            # ----------------------------------
            # 8. Mask classification
            # ----------------------------------

            prediction = mask_model.predict(
                face,
                verbose=0
            )


            # ----------------------------------
            # 9. Convert prediction to label
            # ----------------------------------

            if prediction[0][0] >= 0.5:

                label = "No Mask"

            else:

                label = "Mask"


            # ----------------------------------
            # 10. Store result
            # ----------------------------------

            predictions.append({

                "label": label,

                "confidence": float(
                    prediction[0][0]
                ),

                "bounding_box": {

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                }

            })


    # --------------------------------------
    # 11. Return results
    # --------------------------------------

    return {
        "predictions": predictions
    }