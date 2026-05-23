<p align="center">

# 🚗 vehicle-tracking-yolov8

#### 🚀 Real-Time Vehicle Tracking & Counting using YOLOv8 + ByteTrack

</p>

<p align="center">
<a href="#"><img src="https://img.shields.io/badge/Open%20in%20Colab-orange?logo=google-colab" alt="Open in Colab"></a>
<img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
<img src="https://img.shields.io/badge/YOLOv8-v8.x-purple?logo=google-cloud" alt="YOLOv8">
<img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" alt="OpenCV">
<img src="https://img.shields.io/badge/Ultralytics-v8-red" alt="Ultralytics">
</p>

---

<!-- DEVELOPER -->
<p align="center">
  <img src="https://img.shields.io/badge/Developer-V.%20Ashish%20Nadh-lightgrey">
  <img src="https://img.shields.io/badge/Institution-GITAM%20University%20Visakhapatnam-informational">

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
- Detects and classifies only **Cars, Buses, and Trucks**
- Assigns a **unique tracking ID** per vehicle across frames
- Uses a **spatial crossing line** to count each vehicle **exactly once**

---

## 🧠 System Architecture

```text
Video Input (car-detection.mp4)
          │
          ▼
┌─────────────────────────────────┐
│    OpenCV Frame Reader          │
│ • Reads frames at video FPS     │
│ • Extracts frame dimensions     │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│   YOLOv8n Detection Model       │
│ • COCO filtered: Car=2,         │
│   Bus=5, Truck=7                │
│ • Anchor-free C2f backbone      │
│ • PANet neck for multi-scale    │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ ByteTrack Tracker               │
│ • Motion prediction + IoU match │
│ • Persistent IDs across frames  │
│ • Handles occlusion gracefully  │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Spatial Crossing Line Logic     │
│ • Virtual line across lanes     │
│ • Triggers count on crossing    │
│ • Prevents double count         │
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
| **Computer Vision** | OpenCV (`cv2`) |
| **GPU Runtime** | Google Colab (T4 GPU) |
| **Development** | Google Colab Notebooks |

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

### 2. Set Up Dependencies
```bash
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0
```

### 3. Run the Detection Script
```bash
python src/detect_vehicles.py --input car-detection.mp4 --output results/output_counted.mp4
```

### 4. View Results
The output video will be saved in the `results/` folder with:
- Bounding boxes drawn in **class-specific colors** (Car: Green, Bus: Blue, Truck: Red)
- Unique **Track IDs** displayed on each vehicle
- Running **vehicle count** overlay in top-left corner
- **Counting line** visualization for tracking accuracy

---

## 📊 Results

### Output Example
The system produces an annotated video with:
- Bounding boxes around detected vehicles (color-coded by class)
- Unique Track IDs displayed on each vehicle
- Running vehicle count displayed on-screen
- Custom class filter applied (Cars ✅, Buses ✅, Trucks ✅, Others ❌)

---

## 💡 Key Project Highlights
- ✅ **No training required** — uses pretrained YOLOv8n weights on COCO
- ✅ **Real-time inference** with persistent vehicle tracking
- ✅ **Accurate counting** using spatial line-crossing logic
- ✅ **GPU-accelerated** on Google Colab T4
- ✅ **Scalable architecture** — easily extendable to other vehicle classes
- ✅ **Production-ready code** — clean, modular, and documented

---

## 📁 Repository Structure
```
vehicle-tracking-yolov8/
├── README.md                     # This file
├── .gitignore                    # Ignores .mp4, .pt files (too large for GitHub)
├── src/
│   └── detect_vehicles.py        # Standalone Python script
└── results/
    └── README.md                 # Output video frames & accuracy results
```

---

## 📖 References
- **YOLOv8**: Ultralytics, *[YOLOv8 Documentation](https://docs.ultralytics.com/)*
- **COCO Dataset**: Lin et al., *[Microsoft COCO: Common Objects in Context](https://cocodataset.org/)*, ECCV 2014
- **ByteTrack**: Zhang et al., *[ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://github.com/ifzhang/ByteTrack)*, ECCV 2022
- **Intel IoT DevKit**: Traffic Video Dataset

---

## 📚 MLSAVA Case Study
This project was developed as part of the **MLSAVA (Machine Learning for Speech & Video Analysis)** case study at **GITAM University, Visakhapatnam**. The case study required implementing a complete computer vision pipeline for real-time object detection, tracking, and counting on video streams.

---

## 🔗 Related Projects
Explore my other AI/ML projects on GitHub:
- 🚦 **[Traffic Sign Recognition using CNN](https://github.com/Ashish3692003/traffic-sign-recognition-cnn)** — Deep Learning project for road sign classification
- 🧠 **[MNIST CNN on FPGA-Zedboard](https://github.com/Ashish3692003/MNIST-CNN-FPGA-Zedboard)** — CNN hardware acceleration on Xilinx FPGA

---

<p align="center">
Made with ❤️ by <strong>V. Ashish Nadh</strong> | <a href="https://github.com/Ashish3692003">View my GitHub</a>
</p>
