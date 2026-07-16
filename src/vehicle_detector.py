"""
Vehicle Detector Module
-----------------------
Uses YOLOv8 to detect vehicles (cars, trucks, buses) in video frames.
Returns 2D bounding boxes with confidence scores.

Implemented using modern YOLOv8 instead of older Caffe-based networks.
"""

import numpy as np


class VehicleDetector:
    """Detects vehicles in frames using YOLOv8."""

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {2: 'car', 5: 'bus', 7: 'truck'}

    def __init__(self, model_size='yolov8m', confidence_threshold=0.35):
        """
        Initialize the vehicle detector.

        Args:
            model_size: YOLOv8 model variant ('yolov8n', 'yolov8s', 'yolov8m')
            confidence_threshold: Minimum detection confidence
        """
        from ultralytics import YOLO
        self.model = YOLO(f'{model_size}.pt')
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Detect vehicles in a single frame.

        Args:
            frame: BGR image (numpy array from cv2)

        Returns:
            List of dicts, each with keys:
                'bbox': [x1, y1, x2, y2] pixel coordinates
                'confidence': float
                'class_name': str ('car', 'bus', 'truck')
                'class_id': int
        """
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())

                # Only keep vehicle classes
                if cls_id not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].item()

                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_name': self.VEHICLE_CLASSES[cls_id],
                    'class_id': cls_id,
                })

        return detections
