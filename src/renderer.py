"""
3D Bounding Box Renderer Module
---------------------------------
Draws projected 3D bounding boxes onto video frames.

The rendering style draws front faces in green, rear faces in red,
and connecting edges in the category color. We replicate this visual
style in our custom implementation.
"""

import cv2
import numpy as np


# Edge connectivity for the 3D bounding box
# Corner order: FBL(0), FBR(1), RBR(2), RBL(3), FTL(4), FTR(5), RTR(6), RTL(7)
FRONT_EDGES = [(4, 5), (5, 1), (0, 1), (0, 4)]   # Front face (green)
REAR_EDGES  = [(2, 3), (7, 3), (7, 6), (6, 2)]    # Rear face (red)
SIDE_EDGES  = [(4, 7), (5, 6), (1, 2), (0, 3)]    # Connecting edges (blue)


def hex_to_bgr(hex_color):
    """Convert hex color string to BGR tuple for OpenCV."""
    hex_color = hex_color.strip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


# Category colors matching the reference thesis style
CATEGORY_COLORS = {
    'car':   hex_to_bgr('3399FF'),  # Blue
    'bus':   hex_to_bgr('FF9933'),  # Orange
    'truck': hex_to_bgr('FF33CC'),  # Pink
    'motorcycle': hex_to_bgr('FFFF00'), # Yellow
}

# Fixed face colors (matching reference: front=green, rear=red)
FRONT_COLOR = (0, 255, 0)    # Green
REAR_COLOR  = (0, 0, 255)    # Red


def draw_3d_bbox(frame, corners_2d, vehicle_type='car', confidence=1.0,
                 line_thickness=2, show_label=True, depth=None, track_id=None):
    """
    Draw a 3D bounding box wireframe on the frame.

    The visual style matches the reference thesis:
    - Front face edges in GREEN
    - Rear face edges in RED
    - Side/connecting edges in the category COLOR (blue for cars)

    Args:
        frame: BGR image (modified in-place)
        corners_2d: 8x2 array of projected 2D pixel coordinates
        vehicle_type: 'car', 'bus', or 'truck'
        confidence: Detection confidence score
        line_thickness: Thickness of drawn lines
        show_label: Whether to draw text label
        depth: Estimated depth for label display
    """
    pts = corners_2d.astype(int)
    category_color = CATEGORY_COLORS.get(vehicle_type, CATEGORY_COLORS['car'])

    # Check if all points are within a reasonable range
    h, w = frame.shape[:2]
    margin = 500
    valid = True
    for pt in pts:
        if pt[0] < -margin or pt[0] > w + margin or pt[1] < -margin or pt[1] > h + margin:
            valid = False
            break
    if not valid:
        return

    # Draw front face edges (green) - matching reference thesis style
    for i, j in FRONT_EDGES:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), FRONT_COLOR, line_thickness)

    # Draw rear face edges (red)
    for i, j in REAR_EDGES:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), REAR_COLOR, line_thickness)

    # Draw connecting side edges (category color)
    for i, j in SIDE_EDGES:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), category_color, line_thickness)

    if show_label:
        label_pt = (pts[4][0], pts[4][1] - 8)
        if track_id is not None:
            label_text = f'{vehicle_type} #{track_id}'
        else:
            label_text = f'{vehicle_type}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Draw background rectangle for text
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(frame,
                      (label_pt[0], label_pt[1] - th - 4),
                      (label_pt[0] + tw, label_pt[1] + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, label_text, label_pt, font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def draw_info_overlay(frame, frame_num, total_frames, num_detections, fps=0):
    """
    Draw an information overlay on the frame.

    Args:
        frame: BGR image (modified in-place)
        frame_num: Current frame number
        total_frames: Total number of frames
        num_detections: Number of detections in this frame
        fps: Processing FPS
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Info text
    info = f'Frame: {frame_num}/{total_frames}  |  Vehicles: {num_detections}'
    if fps > 0:
        info += f'  |  FPS: {fps:.1f}'
    cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Removed title as per user request
