import cv2
import numpy as np

from constants import *

class SignTracker:
    def __init__(self, max_lost=MAX_LOST, iou_threshold=IOU_THRESHOLD, max_shift=50):
        self.trackers = {}
        self.next_id = 0
        self.lost_counts = {}
        self.labels = {}
        self.scores = {}
        self.prev_boxes = {}
        self.updated_boxes = {}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.max_shift = max_shift
        self._last_frame = None

    def set_last_frame(self, frame):
        self._last_frame = frame

    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0

    def is_bbox_mostly_outside(self, bbox, frame_shape, max_out_ratio=0.3):
        x, y, w, h = map(int, bbox)
        frame_h, frame_w = frame_shape[:2]

        bbox_area = w * h
        if bbox_area == 0:
            return True

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        inside_area = max(0, x2 - x1) * max(0, y2 - y1)
        outside_area = bbox_area - inside_area

        return outside_area / bbox_area > max_out_ratio
        
    def is_bbox_resized_abnormally(self, old_bbox, new_bbox, scale_thresh=2.5):
        _, _, w1, h1 = old_bbox
        _, _, w2, h2 = new_bbox
        area1 = w1 * h1
        area2 = w2 * h2

        if area1 == 0 or area2 == 0:
            return True

        scale = area2 / area1
        return scale > scale_thresh or scale < 1 / scale_thresh
    
    def is_meaningful_patch(self, frame, bbox, threshold=15):
        x, y, w, h = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        patch = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > threshold
    
    def is_inside_or_outside_existing_box(self, new_box, margin=10):
        for tid, tracker in self.trackers.items():
            success, tracked_box = tracker.update(self._last_frame)
            if not success:
                continue
            x1, y1, w1, h1 = tracked_box
            x2, y2, w2, h2 = new_box

            left1, top1, right1, bottom1 = x1, y1, x1 + w1, y1 + h1
            left2, top2, right2, bottom2 = x2, y2, x2 + w2, y2 + h2

            if (left2 >= left1 - margin and top2 >= top1 - margin and
                right2 <= right1 + margin and bottom2 <= bottom1 + margin):
                return True
            if (left1 >= left2 - margin and top1 >= top2 - margin and
                right1 <= right2 + margin and bottom1 <= bottom2 + margin):
                return True
        return False

    def is_large_shift(self, old_bbox, new_bbox):
        x1, y1, w1, h1 = old_bbox
        x2, y2, w2, h2 = new_bbox
        dx = abs((x2 + w2 / 2) - (x1 + w1 / 2))
        dy = abs((y2 + h2 / 2) - (y1 + h1 / 2))
        return dx > self.max_shift or dy > self.max_shift

    def add_tracker(self, frame, bbox, label, score):
        self.set_last_frame(frame)
        updated_existing = False

        for tid, tracker in self.trackers.items():
            success, tracked_box = tracker.update(self._last_frame)
            if not success:
                continue

            iou = self._iou(bbox, tracked_box)
            if iou > self.iou_threshold:
                if score > self.scores.get(tid, 0):
                    new_tracker = cv2.TrackerCSRT_create()
                    new_tracker.init(frame, bbox)
                    self.trackers[tid] = new_tracker
                    self.labels[tid] = label
                    self.scores[tid] = score
                    self.prev_boxes[tid] = bbox
                updated_existing = True
                break

        if not updated_existing and not self.is_inside_or_outside_existing_box(bbox):
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            tid = self.next_id
            self.trackers[tid] = tracker
            self.lost_counts[tid] = 0
            self.labels[tid] = label
            self.scores[tid] = score
            self.prev_boxes[tid] = bbox
            self.next_id += 1

    def update(self, frame):
        results = []
        remove_ids = []
        self.updated_boxes = {}

        for tid, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success and self.is_meaningful_patch(frame, bbox):
                self.updated_boxes[tid] = bbox
            else:
                self.lost_counts[tid] = self.lost_counts.get(tid, 0) + 1
                if self.lost_counts[tid] > self.max_lost:
                    remove_ids.append(tid)

        for tid, bbox in self.updated_boxes.items():
            if tid not in self.prev_boxes:
                continue

            if (
                self.is_large_shift(self.prev_boxes[tid], bbox) or
                self.is_bbox_resized_abnormally(self.prev_boxes[tid], bbox) or
                self.is_bbox_mostly_outside(bbox, frame.shape)
            ):
                remove_ids.append(tid)
                continue

            self.prev_boxes[tid] = bbox
            x, y, w, h = map(int, bbox)
            results.append((x, y, w, h, tid, self.labels[tid], self.scores[tid]))
            self.lost_counts[tid] = 0

        for tid in set(remove_ids):
            self.trackers.pop(tid, None)
            self.lost_counts.pop(tid, None)
            self.labels.pop(tid, None)
            self.scores.pop(tid, None)
            self.prev_boxes.pop(tid, None)
            self.updated_boxes.pop(tid, None)

        return results
