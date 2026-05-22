# 📂 Dataset

## Traffic Video Dataset

This project uses the **Intel IoT DevKit traffic video** as the input video stream.

### Dataset Details

| Property | Details |
|---|---|
| **Filename** | `car-detection.mp4` |
| **Source** | [Intel IoT DevKit](https://www.intel.ai/iot-devkit/) |
| **Type** | Continuous RGB video (.mp4) |
| **Perspective** | Overhead multi-lane highway (CCTV-style) |
| **Content** | Cars, buses, and trucks moving through traffic lanes |

### How to Use

1. Download the video from the [Intel IoT DevKit dataset](https://www.intel.ai/iot-devkit/)
2. Save it inside this `data/` folder with the name `car-detection.mp4`
3. Run the model: `python src/detect_vehicles.py --input data/car-detection.mp4`

> ⚠️ Video files (`.mp4`) are **not tracked by Git** (see `.gitignore`). Place your video manually in this folder after cloning the repo.

### Alternative Datasets

You can replace this video with **any overhead highway/CCTV traffic footage**:

- **UA-DETRAC** — Multi-object tracking dataset for vehicle detection
- **CityFlow** — Large-scale multi-target multi-camera tracking dataset
- **KITTI Vision Benchmark** — Street-view traffic dataset

As long as the video is **overhead-view** (top-down) of a highway/road, the YOLOv8n model will detect vehicles and the ByteTrack will assign persistent IDs for counting.
