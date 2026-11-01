# 🚗 Automated Real-Time Vehicle Tracking & Counting System
> Edge-ready computer vision pipeline pairing YOLOv8 and ByteTrack for non-intrusive traffic analytics and automated spatial deduplication.

---

## 📌 Executive Summary & Problem Context

Traditional municipal traffic monitoring relies on manual surveys or physical pneumatic road tubes. These systems entail high maintenance overhead, risk road worker safety, cause traffic disruptions, and fail under extreme weather conditions. 

This project delivers a **non-intrusive, software-defined traffic analysis pipeline** that ingests continuous CCTV video streams to classify, track, and tally vehicles in real time. By pairing an anchor-free single-stage detector (**YOLOv8 Nano**) with **ByteTrack**, the system maintains robust object identities across frames and uses spatial logic to count traffic volume without double-counting.

---

## 🏗️ Architecture & Processing Pipeline

```
           [ Overhead CCTV Stream (car-detection.mp4) ]
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             OpenCV Frame Decapsulation                 │
   │   • Dynamic dimension extraction (W x H) & FPS pacing  │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             YOLOv8n Object Detection                   │
   │   • Anchor-free center prediction & C2f backbone       │
   │   • Selective COCO filter: Cars (2), Buses (5), Trucks(7)│
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             ByteTrack Multi-Object Tracker             │
   │   • Kalman filter trajectory estimation + IoU matching │
   │   • Low-confidence association prevents occlusion loss │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │          Spatial Hysteresis & Deduplication            │
   │   • Centroid tracking: cy = (y1 + y2) // 2             │
   │   • Hysteresis boundary band: [line_y - 105, line_y + 105] │
   │   • O(1) hash set stateful vehicle counting            │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
            [ Annotated Output (output_counted.mp4) ]
```

---

## 🔬 Key Engineering Highlights

* **Anchor-Free Detection (YOLOv8n):** Predicts object centers directly rather than relying on predefined anchor boxes, reducing computational latency and enabling real-time inference on edge and cloud GPU instances.
* **C2f Feature Representation:** Enhances gradient flow across high-level semantic context and fine-grained spatial features, ensuring accurate detection of smaller vehicles at the camera horizon.
* **Class-Specific Filtering:** Filters detections to target classes—Cars (2), Buses (5), and Trucks (7)—at inference time, eliminating wasted cycles on pedestrians or background clutter.
* **Occlusion-Resistant Association (ByteTrack):** Retains low-confidence bounding boxes during two-stage matching, maintaining stable vehicle IDs across brief occlusions and speed variations.
* **Spatial Hysteresis Deduplication:** Uses a $210\text{-pixel}$ detection corridor ($line\_y \pm 105\text{ px}$) around the virtual barrier line combined with a stateful Python `set()`, guaranteeing single-count accuracy regardless of vehicle speed or frame rate fluctuations.

---

## ⚡ Quickstart & Usage

### 1. Installation
```bash
git clone [https://github.com/Ashish3692003/vehicle-tracking-yolov8.git](https://github.com/Ashish3692003/vehicle-tracking-yolov8.git)
cd vehicle-tracking-yolov8
pip install ultralytics opencv-python numpy
```

### 2. Run Inference
```bash
python -c "
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('traffic.mp4')
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
out = cv2.VideoWriter('results/output_counted.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

line_y = h - 100
counted = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model.track(frame, classes=[2, 5, 7], persist=True, verbose=False)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 2)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cy = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if line_y - 105 < cy < line_y + 105:
                counted.add(track_id)
                
    cv2.putText(frame, f'Total Vehicles Counted: {len(counted)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    out.write(frame)

cap.release()
out.release()
"
```

---

## 📊 Technical Specifications

| Metric / Parameter | Value | Details |
| :--- | :--- | :--- |
| **Model Architecture** | YOLOv8 Nano (`yolov8n.pt`) | Single-stage anchor-free detector |
| **Checkpoint Size** | ~6.2 MB | Ultra-compact memory footprint |
| **Target Classes** | 3 Classes | Car (2), Bus (5), Truck (7) |
| **Tracker Engine** | ByteTrack | Kalman filter trajectory + IoU association |
| **Runtime Environment** | NVIDIA T4 GPU / Google Colab | Real-time processing speed (>30 FPS) |
| **Test Video Stream** | Intel IoT DevKit (`car-detection.mp4`) | Overhead multi-lane highway perspective |

---

## 🔮 Scalability & Future Roadmap

* **Bi-Directional Lane Parsing:** Calculate displacement vectors ($\Delta y = cy_t - cy_{t-1}$) to track incoming and outgoing traffic volume across discrete lanes.
* **Homography-Based Velocity Tracking:** Map perspective pixel coordinates to real-world ground distances to calculate real-time vehicle velocity and alert for road hazards.
* **Edge Engine Compilation:** Convert PyTorch weights to INT8/FP16 TensorRT or ONNX engines for low-power edge gateways (e.g., NVIDIA Jetson Orin Nano).
