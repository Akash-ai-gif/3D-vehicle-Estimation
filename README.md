# 3D Vehicle Estimation & Tracking

This project is a modern, robust pipeline for **3D Vehicle Pose Estimation and Tracking** using a monocular camera. It is capable of taking a standard 2D video and computing the 3D bounding boxes, depth, rotation (yaw), and temporal tracking of vehicles in the scene.

## Key Features
- **YOLOv8 Detection:** Leverages state-of-the-art YOLOv8 for highly accurate, real-time 2D vehicle detection (cars, trucks, buses).
- **Geometric 3D Pose Estimation:** Calculates the distance (Z-depth) and 3D coordinates relative to the camera center without requiring stereo vision or LiDAR.
- **Temporal Object Tracking:** Uses an Intersection-over-Union (IoU) tracker to persistently track vehicles across video frames, assigning them unique IDs.
- **EMA Smoothing:** Employs Exponential Moving Average (EMA) mathematical smoothing to lock down bounding box dimensions and center positions, completely eliminating jitter and lag.
- **Velocity-Driven Yaw:** Intelligently computes the rotation of the vehicle based on 2D visual cues and geometric perspective projection.

## Installation

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: Requires `ultralytics`, `opencv-python`, and `numpy`)*

## Usage

You can run the pipeline on any video using `src/main.py`.

```bash
python src/main.py --input traffic_video.mp4 --output output.avi --fps 30.0
```

### Converting to Playable MP4
Because OpenCV generates uncompressed or raw `.avi` files that aren't playable in modern web browsers, use `ffmpeg` to encode the output to H.264:
```bash
ffmpeg -y -i output.avi -c:v libx264 -preset fast -crf 22 final_output.mp4
```

## Structure
- `src/main.py`: The core pipeline entry point.
- `src/vehicle_detector.py`: Handles YOLOv8 inference.
- `src/pose_estimator.py`: Calculates depth, geometry, and constructs the 3D box.
- `src/tracker.py`: Tracks vehicles temporarily via IoU frame matching.
- `src/renderer.py`: Handles drawing the 3D projection onto the 2D video frames.

## Dataset Support
This codebase has been rigorously tested against standard traffic footage as well as the **KITTI Vision Benchmark Suite**, demonstrating robust stability on thousands of consecutive frames.