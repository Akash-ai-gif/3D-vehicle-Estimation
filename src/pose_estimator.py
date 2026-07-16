"""
3D Pose Estimator Module
-------------------------
Estimates 3D orientation and dimensions of detected vehicles from their
2D bounding boxes. Uses geometric constraints and learned priors based
on the KITTI dataset statistics.

The approach uses geometric heuristics to reconstruct 3D bounding boxes from
projected 2D corners and ground plane equations. Here we adapt this
concept to work without explicit camera calibration by using reasonable
defaults.
"""

import numpy as np


# Average vehicle dimensions from the KITTI dataset (in meters)
# These serve as priors for 3D box estimation
VEHICLE_DIMENSIONS = {
    'car':   {'height': 1.52, 'width': 1.63, 'length': 3.88},
    'bus':   {'height': 3.20, 'width': 2.55, 'length': 10.0},
    'truck': {'height': 3.00, 'width': 2.50, 'length': 6.50},
}


class PoseEstimator:
    """
    Estimates 3D pose (position, orientation, dimensions) of vehicles
    from their 2D bounding box detections.
    """

    def __init__(self, frame_width, frame_height, focal_length=None):
        """
        Initialize the pose estimator with camera parameters.

        Args:
            frame_width: Width of the video frame in pixels
            frame_height: Height of the video frame in pixels
            focal_length: Focal length in pixels (estimated if None)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Estimate focal length if not provided
        # A reasonable approximation for dashcam: focal_length ~ frame_width
        if focal_length is None:
            self.focal_length = frame_width * 0.8
        else:
            self.focal_length = focal_length

        # Build intrinsic camera matrix K
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        self.K = np.array([
            [self.focal_length, 0,                cx],
            [0,                 self.focal_length, cy],
            [0,                 0,                 1 ],
        ], dtype=np.float64)

        # Track history for EMA smoothing
        self.history = {}
        # EMA alpha controls smoothing (0.0 to 1.0). Lower = smoother but more lag.
        # Set to 0.7 to heavily prefer the new detection and minimize tracking lag!
        self.ema_alpha = 0.7

    def estimate_depth(self, bbox, vehicle_type='car'):
        """
        Estimate depth (distance from camera) using the 2D bounding box
        height and known average vehicle dimensions.

        The idea: real_height / depth = pixel_height / focal_length
        => depth = real_height * focal_length / pixel_height

        Args:
            bbox: [x1, y1, x2, y2]
            vehicle_type: 'car', 'bus', or 'truck'

        Returns:
            Estimated depth in meters
        """
        dims = VEHICLE_DIMENSIONS.get(vehicle_type, VEHICLE_DIMENSIONS['car'])
        pixel_height = bbox[3] - bbox[1]

        if pixel_height < 10:
            return 100.0  # Very far away, avoid division issues

        depth = (dims['height'] * self.focal_length) / pixel_height
        return max(depth, 2.0)  # Minimum 2m depth

    def estimate_orientation(self, bbox):
        """
        Estimate the yaw angle (rotation around Y axis) of the vehicle
        based on its position in the image.

        Vehicles near the center tend to face away from camera (yaw ~ 0),
        vehicles on the sides show more of their side profile.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            Estimated yaw angle in radians
        """
        center_x = (bbox[0] + bbox[2]) / 2.0
        # Normalized position: -1 (left) to +1 (right)
        norm_x = (center_x - self.frame_width / 2.0) / (self.frame_width / 2.0)

        # Aspect ratio gives information about orientation
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect = w / max(h, 1)

        # Heuristic: wider boxes (larger aspect ratio) indicate
        # a more side-on view of the vehicle
        if aspect > 1.8:
            # Very wide = near side view
            base_angle = np.pi / 2.0 * np.sign(norm_x) if abs(norm_x) > 0.1 else 0.0
        else:
            # Narrower = more rear/front view
            base_angle = norm_x * np.pi / 4.0

        return base_angle

    def estimate_3d_bbox(self, bbox, vehicle_type='car', track_id=None):
        """
        Estimate the full 3D bounding box for a detected vehicle.

        Inspired by the reconstruct_bb3d method in the reference repo's
        PGP class, but using monocular depth estimation instead of
        explicit ground plane + projection matrix reconstruction.

        Args:
            bbox: [x1, y1, x2, y2]
            vehicle_type: 'car', 'bus', or 'truck'
            track_id: Optional tracking ID for temporal smoothing

        Returns:
            dict with:
                'corners_3d': 8x3 array of 3D corner coordinates
                'center_3d': [X, Y, Z] center position
                'dimensions': [h, w, l]
                'yaw': rotation angle in radians
                'depth': estimated depth
        """
        dims = VEHICLE_DIMENSIONS.get(vehicle_type, VEHICLE_DIMENSIONS['car'])
        h, w, l = dims['height'], dims['width'], dims['length']

        # Estimate depth and orientation
        depth = self.estimate_depth(bbox, vehicle_type)
        yaw = self.estimate_orientation(bbox)

        # Estimate 3D center position
        center_x_px = (bbox[0] + bbox[2]) / 2.0
        center_y_px = (bbox[1] + bbox[3]) / 2.0

        # Back-project center to 3D
        X = (center_x_px - self.K[0, 2]) * depth / self.K[0, 0]
        Y = (center_y_px - self.K[1, 2]) * depth / self.K[1, 1]
        Z = depth

        center_3d = np.array([X, Y, Z])

        # Apply EMA smoothing if tracking ID is provided
        if track_id is not None:
            if track_id in self.history:
                prev = self.history[track_id]
                alpha = self.ema_alpha
                
                # Smooth center position (Lightly, to prevent lag)
                center_3d = alpha * center_3d + (1 - alpha) * prev['center_3d']
                depth = center_3d[2]
                
                # Smooth yaw slightly, taking care of angular wraparound (-pi to pi)
                diff = yaw - prev['yaw']
                # Normalize angle difference to [-pi, pi]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                yaw = prev['yaw'] + alpha * diff
                # Normalize final yaw back to [-pi, pi]
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

            # Update history (only store what needs smoothing)
            self.history[track_id] = {
                'center_3d': center_3d,
                'yaw': yaw
            }

        # Generate 3D bounding box corners (8 corners of a cuboid)
        # Convention: Y is up in camera frame, so height goes in -Y
        corners_3d = self._create_box_corners(center_3d, h, w, l, yaw)

        return {
            'corners_3d': corners_3d,
            'center_3d': center_3d,
            'dimensions': [h, w, l],
            'yaw': yaw,
            'depth': depth,
        }

    def _create_box_corners(self, center, h, w, l, yaw):
        """
        Create the 8 corners of a 3D bounding box.

        The corners are ordered as:
            FBL, FBR, RBR, RBL, FTL, FTR, RTR, RTL
        (Front-Bottom-Left, Front-Bottom-Right, etc.)

        This matches the convention used in the reference thesis code.

        Args:
            center: [X, Y, Z] 3D center position
            h, w, l: height, width, length of the vehicle
            yaw: rotation angle around Y axis

        Returns:
            8x3 numpy array of corner coordinates
        """
        # Rotation matrix around Y axis (inspired by R3x3_y in geometry.py)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([
            [ cos_yaw, 0, sin_yaw],
            [ 0,       1, 0      ],
            [-sin_yaw, 0, cos_yaw],
        ])

        # Half dimensions
        hw, hh, hl = w / 2.0, h / 2.0, l / 2.0

        # 8 corners in local frame (centered at geometric center of vehicle)
        # Order: FBL, FBR, RBR, RBL, FTL, FTR, RTR, RTL
        local_corners = np.array([
            [-hw,  hh,  hl],   # FBL - Front Bottom Left
            [ hw,  hh,  hl],   # FBR - Front Bottom Right
            [ hw,  hh, -hl],   # RBR - Rear Bottom Right
            [-hw,  hh, -hl],   # RBL - Rear Bottom Left
            [-hw, -hh,  hl],   # FTL - Front Top Left
            [ hw, -hh,  hl],   # FTR - Front Top Right
            [ hw, -hh, -hl],   # RTR - Rear Top Right
            [-hw, -hh, -hl],   # RTL - Rear Top Left
        ])

        # Rotate and translate to world coordinates
        rotated = (R @ local_corners.T).T
        corners_3d = rotated + center

        return corners_3d

    def project_3d_to_2d(self, corners_3d):
        """
        Project 3D corners to 2D image coordinates using camera intrinsics.

        Inspired by project_X_to_x in the reference geometry.py:
            x_3xn = P * X_4xn
            x_2xn = x_3xn[0:2] / x_3xn[2]

        Here we simplify since we assume identity extrinsics (P = K[I|0]).

        Args:
            corners_3d: Nx3 array of 3D points

        Returns:
            Nx2 array of 2D pixel coordinates
        """
        # Project: x = K * X (assuming no extrinsic rotation/translation)
        points_3d = corners_3d.T  # 3xN
        projected = self.K @ points_3d  # 3xN

        # Normalize by depth (z coordinate)
        z = projected[2, :]
        # Avoid division by zero
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)

        points_2d = projected[:2, :] / z  # 2xN
        return points_2d.T  # Nx2
