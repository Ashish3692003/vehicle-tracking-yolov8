<p align="center">
  
# 🚗 vehicle-tracking-yolov8
### 🎯 Real-Time Vehicle Tracking & Counting using YOLOv8 + ByteTrack
  
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Open%20in-Colab-orange?logo=google-colab" alt="Open in Colab"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-v8.x-purple?logo=google-cloud" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Ultralytics-v8-red" alt="Ultralytics">
  <img src="https://img.shields.io/badge/MIT-License-yellow" alt="License">
</p>

<!-- DEVELOPER -->
<p align="center">
  <image src="https://img.shields.io/badge/Developer-V.%20Ashish%20Nadh-lightgrey">
  <image src="https://img.shields.io/badge/Institution-GITAM%20University%20Visakhapatnam-informational">
  <image src="https://img.shields.io/badge/Course-MLSAVA%20--%20ML%20for%20Speech%20%26%20Video%20Analysis-blue">
</p>

---

## 📌 Overview

This project implements a **Real-Time Vehicle Tracking and Counting System** on video streams using the **YOLOv8 object detection model** and **ByteTrack multi-object tracker**. The system processes CCTV-style overhead highway footage to detect, track, and count vehicles (Cars, Buses, Trucks) in real-time — providing an accurate, scalable, and non-intrusive alternative to traditional pneumatic road tubes and manual traffic counting.

> Developed as part of the **MLSAVA Case Study** at GITAM University.

---

## 🛣️ Problem Statement

Traditional traffic monitoring relies on:
| Method | Drawbacks |
|---|---|
| Pneumatic Road Tubes | Prone to damage & weather failures |
| Manual Human Counting | Labor-intensive & error-prone |

This project addresses these challenges by deploying a **deep learning-based vision system** that:
- Processes raw video frames **sequentially** in real-time
- Detects and classifies only **Cars, Buses, and Trucks** (ignoring pedestrians, bikes, etc.)
- Assigns a **unique tracking ID** per vehicle across frames (no double-counting)
- Uses a **spatial crossing line** to count each vehicle **exactly once**

---

## 🧠 System Architecture

The full processing pipeline is shown below:

```
[ Input: traffic.mp4 ]
             │
             ▼
┌────────────────────────┐
│   OpenCV Frame Reader  │  ← Extracts frames at video FPS
└───────────┬────────────┘
            │
            ▼
┌─────────────────────────────────┐
│    YOLOv8n Detection Model      │
│  • Backbone: C2f (Coarse-to-Fine)│
│  • Neck: PANet Pyramid          │
│  • Head: Anchor-free predictor  │
│  • Filtered Classes: Car(2),    │
│    Bus(5), Truck(7) from COCO   │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│      ByteTrack Tracker          │
│  • Motion prediction + IoU match│
│  • Persistent IDs across frames │
│  • Handles occlusion gracefully │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│   Spatial Crossing Line Logic   │
│  • Virtual line across lanes    │
│  • Triggers count on crossing   │
│  • Prevents double count        │
└───────────┬─────────────────────┘
            │
            ▼
[ Output: output_counted.mp4 ]
```

---

## 🔧 Technical Details

### Model: YOLOv8 Nano (`yolov8n.pt`)
| Feature | Details |
|---|---|
| **Type** | Anchor-free one-stage detector |
| **Backbone** | C2f Module — captures multi-scale semantic + spatial features |
| **Neck** | PANet (Path Aggregation Network) |
| **Head** | Anchor-free, predicts object center directly |
| **Pretrained On** | COCO dataset (80 object classes) |
| **Filtered COCO IDs** | 2 (car), 5 (bus), 7 (truck) — motorcycles excluded |
| **Performance** | Real-time inference on Google Colab T4 GPU |

### Tracker: ByteTrack
| Feature | Details |
|---|---|
| **Algorithm** | Combines high-score + low-score detections for robust tracking |
| **Idea** | IoU matching + motion prediction between consecutive frames |
| **Output** | Consistent `track_id` assigned to each vehicle across frames |

### Counting Logic
- A **virtual horizontal line** is drawn across the video frame
- When a tracked vehicle's bounding box crosses this line, its count is incremented
- Each `track_id` is counted **only once** per crossing event

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | Ultralytics YOLOv8 (`ultralytics`) |
| **Computer Vision** | OpenCV (cv2) |
| **GPU Runtime** | Google Colab (T4 GPU) |
| **Editing & Development** | Google Colab Notebooks |

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | Intel IoT DevKit — `car-detection.mp4` |
| **Type** | Continuous RGB video stream (.mp4) |
| **Perspective** | Overhead multi-lane highway (CCTV-style view) |
| **Preprocessing** | Frame-by-frame via OpenCV — height, width, FPS extracted dynamically |
| **Training Required?** | ❌ No — uses pretrained YOLOv8n weights on COCO |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ashish3692003/vehicle-tracking-yolov8.git
cd vehicle-tracking-yolov8
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Place the Intel IoT DevKit traffic video in `data/` folder:
```bash
# Download from Intel IoT DevKit website
# Save as: data/car-detection.mp4
```

See `data/README.md` for detailed instructions.

### 4. Run the Model
```bash
python src/detect_vehicles.py --input data/car-detection.mp4 --output results/output_counted.mp4
```

### 5. Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FZRNogE0C9maYdpz13-vdgpGCdaQfm67)

---

## 📁 Repository Structure
```
vehicle-tracking-yolov8/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── .gitignore             # Ignores .mp4, .pt files too large for GitHub
├── src/
│   └── detect_vehicles.py # Standalone Python script (mirrors Colab logic)
├── notebooks/
│   └── Vehicle_Detection.ipynb  # Original Colab notebook (.ipynb)
├── data/
│   └── README.md          # Dataset download instructions
└── results/
    └── README.md          # Output video frames & accuracy results
```

---

## 📊 Key Project Highlights

- ✅ **No training required** — leverages pretrained YOLOv8n on COCO (transfer learning)
- ✅ **Real-time video processing** — frame-by-frame inference at video FPS
- ✅ **Persistent vehicle IDs** — ByteTrack ensures each car keeps its identity
- ✅ **Accurate counting** — spatial crossing line prevents double-counting
- ✅ **Scalable & non-intrusive** — works on any overhead CCTV footage
- ✅ **GPU-accelerated** on Google Colab T4

---

## 🖼️ Output Example

The system produces an annotated video with:
- Bounding boxes around detected vehicles (color-coded by class)
- Unique Track IDs displayed on each vehicle
- Running vehicle count displayed on-screen
- Custom class filter applied (Cars ✅, Buses ✅, Trucks ✅, Others ❌)

---


## 📖 Reference

- YOLOv8: Ultralytics, *YOLOv8 Documentation*, https://docs.ultralytics.com/
- COCO Dataset: Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014
- Intel IoT DevKit: Traffic Video Dataset
- MLSAVA Course Material: GITAM University

---

<p align="center">
  Made with ❤️ by <strong>V. Ashish Nadh</strong> | <a href="https://github.com/Ashish3692003">View my GitHub</a>
</p>
