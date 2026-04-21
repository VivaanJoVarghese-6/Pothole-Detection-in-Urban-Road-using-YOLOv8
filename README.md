# Pothole Detection in Urban Roads using YOLOv8

## Overview

This project implements a pothole detection system using YOLOv8 segmentation. The model is trained on a dataset of road images and is capable of detecting and outlining potholes using both bounding boxes and segmentation masks. The system can be used on images, videos, or live camera input.

---

## Features

* Pothole detection using YOLOv8 segmentation
* Supports image, video, and webcam input
* Outputs bounding boxes and segmentation masks
* Lightweight model (YOLOv8n-seg) for efficient performance

---

## Project Structure

```
pothole-detection/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── runs/
│   └── segment/
│       └── train/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/pothole-detection.git
cd pothole-detection
```

---

### 2. Create Virtual Environment

```
python -m venv yolov8-env
```

Activate the environment:

**Windows (CMD):**

```
yolov8-env\Scripts\activate
```

**Windows (PowerShell):**

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\yolov8-env\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```
pip install --upgrade pip
pip install ultralytics opencv-python torch torchvision torchaudio pandas
```

For GPU support, install PyTorch with CUDA:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset

Download the pothole dataset from Roboflow and place it inside the `data/` directory.

Ensure the structure is:

```
data/
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
└── data.yaml
```

---

## Training

Run the following command to train the model:

```
yolo task=segment mode=train model=yolov8n-seg.pt data=data/data.yaml epochs=50 imgsz=640
```

After training, the model weights will be saved in:

```
runs/segment/train/weights/best.pt
```

---

## Inference

### Run on images

```
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=data/valid/images
```

### Run on video

```
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=path/to/video.mp4
```

### Run on webcam

```
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=0
```

---

## Output

Detection results are saved in:

```
runs/segment/predict/
```

Each output includes:

* Bounding boxes
* Segmentation masks
* Detected pothole regions

---

## Requirements

* Python 3.10 or 3.11
* Ultralytics YOLOv8
* PyTorch (CPU or GPU)
* OpenCV
* Pandas

---

## Notes

* GPU is recommended for faster training
* Ensure correct dataset paths in `data.yaml`
* Use YOLOv8n-seg for better performance on low-resource systems

---

## Submission

Upload your project to GitHub and include:

* Trained weights (`best.pt`)
* Sample output images or videos
* README file

Share your repository link with the required hashtag for evaluation.

---

## References

* Ultralytics YOLOv8 Documentation
* Roboflow Dataset and Tutorials



