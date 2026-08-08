# Real-Time Face Mask Detection

A real-time face mask detection system using **YOLOv8**, **VGG16 Transfer Learning**, and **OpenCV**.

The system first detects faces using YOLOv8 and then classifies each detected face as either **With Mask** or **Without Mask** using a VGG16-based binary classification model.

## Features

- Real-time face detection using YOLOv8
- Face mask classification using VGG16
- Real-time webcam detection using OpenCV
- Image preprocessing and normalization
- Binary classification: Mask / No Mask
- Separate training and inference code

## Project Architecture

```text
Webcam
   ↓
YOLOv8 Face Detection
   ↓
Face Crop
   ↓
Resize to 224 × 224
   ↓
Normalize pixel values
   ↓
VGG16 Classifier
   ↓
Mask / No Mask
```

## Dataset

The project uses a dataset containing approximately **19,345 images**:

- With Mask: 9,608 images
- Without Mask: 9,737 images

The images are resized to **224 × 224** before being passed to the VGG16 model.

The dataset is not included in this repository. Please refer to the original dataset source and its license before downloading or redistributing the data.

## Model

### Face Detection

**YOLOv8** is used to detect faces in the webcam frame.

### Mask Classification

A pretrained **VGG16** model is used as the feature extractor.

The original final classification layer is removed and replaced with:

```python
Dense(1, activation="sigmoid")
```

The classifier predicts:

- `0` → With Mask
- `1` → Without Mask

## Training

The dataset is split into:

- 80% training data
- 20% testing/validation data

Pixel values are normalized from:

```text
[0, 255] → [0, 1]
```

The model is trained using:

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 5
- Input size: 224 × 224 × 3

## Results

The model achieved approximately **93% validation accuracy** on the validation/test split.

> Validation accuracy represents performance on the held-out dataset and does not necessarily represent real-world webcam accuracy.

## Project Structure

```text
FaceMask_Detector/
│
├── data/
│   ├── with_mask/
│   └── without_mask/
│
├── train_model.ipynb
├── webcam.py
├── model.keras
├── model.pt
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FaceMask_Detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training the Model

Open the training notebook:

```bash
jupyter notebook train_model.ipynb
```

Run the notebook cells to train the VGG16 classifier.

The trained model will be saved as:

```text
model.keras
```

## Running Real-Time Detection

After the trained model and YOLO model are available, run:

```bash
python webcam.py
```

The webcam will open and detect faces in real time.

Press:

```text
x
```

to exit the webcam window.

## Technologies Used

- Python
- TensorFlow
- Keras
- VGG16
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## Future Improvements

- Improve performance on faces with spectacles
- Add confidence scores
- Improve real-world generalization
- Add data augmentation
- Evaluate using precision, recall, and F1-score
- Deploy as a web application

## License

This project is intended for educational and portfolio purposes.
