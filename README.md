# American Sign Language Detection

A real-time American Sign Language (ASL) recognition system built using EfficientNet-B0 and MediaPipe. The application detects a user's hand from a webcam feed, extracts the hand region, and classifies it into one of 29 ASL gesture classes.

## Features

* Real-time webcam-based ASL recognition
* Hand detection and tracking using MediaPipe
* EfficientNet-B0 based classifier
* Recognition of A-Z gestures
* Support for `del`, `space`, and `nothing`
* End-to-end deployment pipeline

---

## System Architecture

Webcam Feed

↓

MediaPipe Hand Detection

↓

Hand Bounding Box Extraction

↓

Image Preprocessing (224 × 224)

↓

EfficientNet-B0 Classifier

↓

ASL Prediction

---

## Dataset

Dataset used:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

### Dataset Statistics

* Approximately 87,000 images
* 29 classes
* A-Z alphabets
* Special classes:

  * `del`
  * `space`
  * `nothing`

### Data Split

* Training: 80%
* Validation: 10%
* Test: 10%

---

## Model

### Backbone

EfficientNet-B0

### Training Configuration

* Transfer Learning using ImageNet pretrained weights
* Input Resolution: 224 × 224
* Loss Function: CrossEntropyLoss
* Optimizer: AdamW

---

## Repository Structure

```text
American-Sign-Language-Detection/
│
├── app.py
├── asl_efficientnet_b0.pth
├── classes.json
├── requirements.txt
├── README.md
└── ASL_Training.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/American-Sign-Language-Detection.git
cd American-Sign-Language-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run:

```bash
python app.py
```

A webcam window will open and display the predicted ASL gesture in real time.

Press:

```text
q
```

to exit the application.

---

## Results

The model achieved near-perfect accuracy on the curated dataset. To better assess practical performance, the classifier was integrated with MediaPipe and tested using live webcam input.

The final system successfully recognizes most ASL gestures in real time under normal lighting conditions.

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* MediaPipe
* OpenCV
* Pillow

---

## Conclusion

This project demonstrates an end-to-end computer vision pipeline for real-time American Sign Language recognition. By combining MediaPipe hand tracking with an EfficientNet-B0 classifier, the system performs live gesture recognition directly from webcam input and provides a foundation for more advanced sign language translation systems.
