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
    'motorcycle': {'height': 1.50, 'width': 0.80, 'length': 2.20},
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

    def estimate_3d_bbox(self, bbox, vehicle_type='car', track_id=None):
        """
        Estimate the full 3D bounding box using 2D-3D projection alignment.
        This dynamically scales the 3D box's depth and rotation so its 2D projection
        perfectly matches the dimensions of the YOLO 2D bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
            vehicle_type: 'car', 'bus', or 'truck'
            track_id: Optional tracking ID for temporal smoothing

        Returns:
            dict with 3D properties
        """
        dims = VEHICLE_DIMENSIONS.get(vehicle_type, VEHICLE_DIMENSIONS['car'])
        h, w, l = dims['height'], dims['width'], dims['length']

        x1, y1, x2, y2 = bbox
        cx_2d = (x1 + x2) / 2.0
        cy_2d = (y1 + y2) / 2.0
        w_2d = max(x2 - x1, 1.0)
        h_2d = max(y2 - y1, 1.0)

        # Baseline heuristic yaw to break 180-degree ties
        # (Assuming cars tend to face somewhat away or towards the vanishing point)
        norm_x = (cx_2d - self.frame_width / 2.0) / (self.frame_width / 2.0)
        heuristic_yaw = norm_x * np.pi / 4.0

        best_error = float('inf')
        best_yaw = 0.0
        best_Z = 10.0

        # Grid search over yaw angles from -180 to 180 degrees
        for yaw_deg in range(-180, 180, 5):
            yaw = np.radians(yaw_deg)

            # Test at an arbitrary depth to measure the projected aspect ratio
            Z_test = 10.0
            X_test = (cx_2d - self.K[0, 2]) * Z_test / self.K[0, 0]
            Y_test = (cy_2d - self.K[1, 2]) * Z_test / self.K[1, 1]

            corners_3d = self._create_box_corners(np.array([X_test, Y_test, Z_test]), h, w, l, yaw)
            corners_2d = self.project_3d_to_2d(corners_3d)

            proj_x_min, proj_x_max = np.min(corners_2d[:, 0]), np.max(corners_2d[:, 0])
            proj_y_min, proj_y_max = np.min(corners_2d[:, 1]), np.max(corners_2d[:, 1])

            proj_w = max(proj_x_max - proj_x_min, 1.0)
            proj_h = max(proj_y_max - proj_y_min, 1.0)

            # Calculate required depth to match the YOLO 2D bounding box dimensions
            Z_req_w = Z_test * (proj_w / w_2d)
            Z_req_h = Z_test * (proj_h / h_2d)

            # Error measures aspect ratio mismatch (Z_req_w vs Z_req_h)
            # plus a slight penalty for deviating from heuristic yaw to break symmetry
            angular_diff = abs((yaw - heuristic_yaw + np.pi) % (2 * np.pi) - np.pi)
            error = abs(Z_req_w - Z_req_h) + 0.5 * angular_diff

            if error < best_error:
                best_error = error
                best_yaw = yaw
                best_Z = (Z_req_w + Z_req_h) / 2.0

        depth = best_Z
        yaw = best_yaw
        X = (cx_2d - self.K[0, 2]) * depth / self.K[0, 0]
        Y = (cy_2d - self.K[1, 2]) * depth / self.K[1, 1]
        
        center_3d = np.array([X, Y, depth])

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
