import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

class IouTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.tracks = {}  # dict mapping track_id -> {'bbox': box, 'type': type, 'lost': count}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, detections):
        """
        Update tracker with new detections.
        detections: list of (bbox, vehicle_type, conf)
        Returns: list of (track_id, bbox, vehicle_type, conf)
        """
        updated_tracks = []
        unmatched_detections = list(detections)

        # Match existing tracks to new detections
        for track_id, track_data in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_match_idx = -1

            for i, (det_bbox, det_type, det_conf) in enumerate(unmatched_detections):
                iou = calculate_iou(track_data['bbox'], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_match_idx >= 0:
                # Match found
                matched_det = unmatched_detections.pop(best_match_idx)
                self.tracks[track_id]['bbox'] = matched_det[0]
                self.tracks[track_id]['type'] = matched_det[1]
                self.tracks[track_id]['conf'] = matched_det[2]
                self.tracks[track_id]['lost'] = 0
                updated_tracks.append((track_id, matched_det[0], matched_det[1], matched_det[2]))
            else:
                # Track lost
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]

        # Add new tracks for unmatched detections
        for det_bbox, det_type, det_conf in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {'bbox': det_bbox, 'type': det_type, 'conf': det_conf, 'lost': 0}
            updated_tracks.append((track_id, det_bbox, det_type, det_conf))

        return updated_tracks
