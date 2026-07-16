"""
3D Vehicle Detection and Pose Estimation - Main Pipeline
----------------------------------------------------------
Processes a raw dashcam video and outputs a new video with
3D bounding boxes rendered around each detected vehicle.

Architecture implemented as a custom, modern Python pipeline using YOLOv8
for detection and geometric heuristics for 3D estimation.

Usage:
    python main.py --input raw_video.mp4 --output result_3d.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vehicle_detector import VehicleDetector
from pose_estimator import PoseEstimator
from renderer import draw_3d_bbox, draw_info_overlay
from tracker import IouTracker


def process_video(input_path, output_path, max_frames=None,
                  confidence=0.35, skip_frames=0, show_labels=True, target_fps=None):
    """
    Main processing pipeline: read video -> detect -> estimate 3D -> render -> write.

    Args:
        input_path: Path to input MP4 video or image sequence (e.g., %10d.png)
        output_path: Path to output video
        max_frames: Maximum number of frames to process (None = all)
        confidence: Minimum detection confidence
        skip_frames: Process every Nth frame (0 = process all)
        show_labels: Whether to show text labels on boxes
        target_fps: Override FPS for the output video (useful for image sequences)
    """
    # -------------------------------------------------------------------------
    # Step 1: Open video or image sequence
    # -------------------------------------------------------------------------
    is_directory = os.path.isdir(input_path)
    image_files = []
    
    if is_directory:
        image_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f'ERROR: No images found in directory: {input_path}')
            sys.exit(1)
        
        # Read first frame to get resolution
        first_frame = cv2.imread(image_files[0])
        frame_height, frame_width = first_frame.shape[:2]
        fps = target_fps if target_fps else 10.0
        total_frames = len(image_files)
        print(f'Input directory: {input_path}')
        
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f'ERROR: Cannot open video or sequence: {input_path}')
            sys.exit(1)

        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if target_fps is not None:
            fps = target_fps
        elif fps <= 0 or '%' in input_path:
            fps = 10.0  # Default to 10 FPS for sequences (like KITTI) if not specified

        # For image sequences, total_frames might return 0 or -1
        if total_frames <= 0 and '%' in input_path:
            print(f'Input sequence: {input_path}')
            total_frames = -1  # Unknown length
        else:
            print(f'Input video/sequence: {input_path}')

    print(f'  Resolution: {frame_width}x{frame_height}')
    print(f'  FPS: {fps:.1f}')
    print(f'  Total frames: {total_frames}')

    # -------------------------------------------------------------------------
    # Step 2: Initialize components
    # -------------------------------------------------------------------------
    print('\nInitializing YOLOv8 vehicle detector...')
    detector = VehicleDetector(model_size='yolov8m',
                                confidence_threshold=confidence)

    print('Initializing 3D pose estimator...')
    estimator = PoseEstimator(frame_width, frame_height)

    print('Initializing Object Tracker...')
    tracker = IouTracker(iou_threshold=0.3, max_lost=5)

    # -------------------------------------------------------------------------
    # Step 3: Setup video writer
    # -------------------------------------------------------------------------
    # Use MP4 codec or fallback to XVID for avi
    if output_path.lower().endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        print(f'ERROR: Cannot create output video: {output_path}')
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 4: Process frames
    # -------------------------------------------------------------------------
    print('\nProcessing video...')
    frame_num = 0
    processed = 0
    start_time = time.time()
    frames_to_process = min(max_frames, total_frames) if max_frames and total_frames > 0 else (max_frames if max_frames else total_frames)
    
    while True:
        if max_frames and frame_num >= max_frames:
            break

        if is_directory:
            if frame_num >= len(image_files):
                break
            frame = cv2.imread(image_files[frame_num])
            if frame is None:
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_num += 1

        # Decide whether to run detection on this frame
        run_detection = (skip_frames == 0) or (frame_num % (skip_frames + 1) == 0)

        if run_detection:
            # ---------------------------------------------------------------------
            # Detection & Tracking
            # ---------------------------------------------------------------------
            # Detect vehicles
            raw_detections = detector.detect(frame)
            
            # Prepare detections for tracker: list of (bbox, vehicle_type, conf)
            tracker_inputs = [(d['bbox'], d['class_name'], d['confidence']) for d in raw_detections]
            
            # Update tracker
            tracked_objects = tracker.update(tracker_inputs)
        else:
            # Update tracker without new detections
            tracked_objects = tracker.update([])

        # ---------------------------------------------------------------------
        # 3D Pose Estimation & Rendering
        # ---------------------------------------------------------------------
        for track_id, bbox, vehicle_type, conf in tracked_objects:
            
            # Estimate 3D bounding box WITH tracking ID
            result = estimator.estimate_3d_bbox(bbox, vehicle_type, track_id=track_id)

            # Project 3D corners to 2D
            corners_2d = estimator.project_3d_to_2d(result['corners_3d'])

            # Draw the 3D bounding box on the frame
            draw_3d_bbox(frame, corners_2d,
                         vehicle_type=vehicle_type,
                         confidence=conf,
                         depth=result['depth'],
                         show_label=show_labels,
                         track_id=track_id)

        # Draw info overlay
        elapsed = time.time() - start_time
        current_fps = frame_num / elapsed if elapsed > 0 else 0
        draw_info_overlay(frame, frame_num, total_frames, 
                          len(tracked_objects), fps=current_fps)

        # Write frame
        writer.write(frame)
        processed += 1

        # Progress reporting
        if frame_num % 50 == 0 or frame_num == 1:
            pct = (frame_num / frames_to_process) * 100
            print(f'  Frame {frame_num}/{frames_to_process} '
                  f'({pct:.1f}%) - {len(tracked_objects)} vehicles detected '
                  f'- {current_fps:.1f} FPS', flush=True)

    # -------------------------------------------------------------------------
    # Step 5: Cleanup
    # -------------------------------------------------------------------------
    if not is_directory:
        cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f'\nDone! Processed {processed} frames in {elapsed:.1f}s')
    print(f'Average speed: {processed / elapsed:.1f} FPS')
    print(f'Output saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='3D Vehicle Detection and Pose Estimation')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input MP4 video (raw, no bounding boxes)')
    parser.add_argument('--output', type=str, default='result_3d_boxes.mp4',
                        help='Path to output MP4 video')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (default: all)')
    parser.add_argument('--confidence', type=float, default=0.35,
                        help='Minimum detection confidence (default: 0.35)')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Skip N frames between detections (default: 0)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Override FPS for output video (useful for image sequences)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Disable text labels on bounding boxes')

    args = parser.parse_args()

    if not os.path.exists(args.input) and '%' not in args.input:
        print(f'ERROR: Input file not found: {args.input}')
        sys.exit(1)

    process_video(
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        confidence=args.confidence,
        skip_frames=args.skip_frames,
        show_labels=not args.no_labels,
        target_fps=args.fps,
    )


if __name__ == '__main__':
    main()
