# Notebooks & Interactive Development

This directory contains Jupyter notebooks and interactive development resources for the vehicle tracking project.

## MLSAVA Case Study Notebook

The complete project implementation is available as a Google Colab notebook:

- **MLSAVA Case Study Notebook**: [MLSAVA_case_study_Ashish_389.ipynb](https://colab.research.google.com/drive/1KrWyRHYO8iDyk57okI3b-BhQXoHMafePhhO6KFFy3O8)

## Project Development Notebook

The hands-on vehicle detection implementation notebook:

- **Vehicle Detection Notebook**: [Vehicle_Detection_Ashish.ipynb](https://colab.research.google.com/drive/1FZRNogE0C9maYdpz13-vdgpGCdaQfm67)

## Notebook Contents

The Colab notebooks cover:

1. **Environment Setup** - Installing Ultralytics YOLO, supervision, and OpenCV
2. **Model Loading** - Loading YOLOv8n pretrained weights
3. **Video Processing** - Frame extraction and processing pipeline
4. **Object Detection** - YOLOv8 inference with custom class filtering
5. **Object Tracking** - ByteTrack integration for multi-object tracking
6. **Spatial Counting** - Horizontal line-based vehicle counting
7. **Visualization** - Bounding boxes, tracking IDs, and counters
8. **Result Export** - Saving processed video and count logs

## Running the Notebooks

### Option 1: Google Colab (Recommended)

Click the notebook links above to open directly in Google Colab. No local setup required.

### Option 2: Jupyter Locally

```bash
# Clone the repository
git clone https://github.com/Ashish3692003/vehicle-tracking-yolov8.git
cd vehicle-tracking-yolov8

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Notebooks in This Project

| Notebook | Purpose | Link |
|----------|---------|------|
| MLSAVA Case Study | Full implementation with documentation | [Open on Colab](https://colab.research.google.com/drive/1KrWyRHYO8iDyk57okI3b-BhQXoHMafePhhO6KFFy3O8) |
| Vehicle Detection | Hands-on detection pipeline | [Open on Colab](https://colab.research.google.com/drive/1FZRNogE0C9maYdpz13-vdgpGCdaQfm67) |

---

### Related Notebooks

- [Traffic Sign Recognition CNN](https://github.com/Ashish3692003/Traffic-Sign-Recognition-CNN) - Image classification using CNNs
- [MNIST CNN on FPGA](https://github.com/Ashish3692003/MNIST-CNN-FPGA-Zedboard) - Hardware-accelerated inference on Xilinx Zedboard
