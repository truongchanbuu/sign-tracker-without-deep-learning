import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from constants import *
from utils import *
from sign_tracker import *

class TrafficSignRecognizer:
    def __init__(self):
        self.frame_count = 0
        self.sign_tracker = SignTracker()
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.recent_checked_boxes = deque(maxlen=50)
        self.templates = {
            "circle": self.load_templates("sign_templates/circle"),
            "triangle": self.load_templates("sign_templates/triangle"),
            "rectangle": self.load_templates("sign_templates/rectangle"),
        }
        
    def load_templates(self, folder_path):
        templates = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                path = os.path.join(folder_path, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                kp, des = self.sift.detectAndCompute(img, None)
                if des is None or len(des) < 2:
                    continue
                
                templates.append({
                    "name": filename,
                    "image": img,
                    "keypoints": kp,
                    "descriptors": des
                })

        return templates
  
    def is_existed(self, new_box, margin=10):
        x2, y2, w2, h2 = new_box
        for x1, y1, w1, h1 in self.recent_checked_boxes:
            left1, top1, right1, bottom1 = x1, y1, x1 + w1, y1 + h1
            left2, top2, right2, bottom2 = x2, y2, x2 + w2, y2 + h2

            if (left2 >= left1 - margin and top2 >= top1 - margin and
                right2 <= right1 + margin and bottom2 <= bottom1 + margin):
                return True
            if (left1 >= left2 - margin and top1 >= top2 - margin and
                right1 <= right2 + margin and bottom1 <= bottom2 + margin):
                return True
        return False
    
    def feature_match_sift(self, roi_gray, shape_type, min_matches=MIN_MATCHES):
        kp2, des2 = self.sift.detectAndCompute(roi_gray, None)
        if des2 is None or len(des2) < 2:
            return None

        best_match = None
        max_good_matches = 0

        for template in self.templates[shape_type]:
            template_img = template["image"]
            kp1 = template["keypoints"]
            des1 = template["descriptors"]
            template_name = template['name']
            
            if des1 is None:
                continue
            
            des1 = np.float32(des1)
            des2 = np.float32(des2)
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.8 * n.distance:
                    good.append(m)
            
            print(f'{self.frame_count}: {template_name} - {len(good)}')
            if len(good) > max_good_matches:
                max_good_matches = len(good)
                best_match = {
                    "template": template_img,
                    "good_matches": good,
                    "kp1": kp1,
                    "kp2": kp2,
                    'name': template_name
                }
                
        if best_match and max_good_matches >= min_matches:
            return best_match
        else:
            return None
        
    def non_max_suppression(self, boxes, overlap_thresh=OVERLAP_THRESH):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)  # sort theo bottom y

        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)

            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:-1]]
            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlap_thresh)[0])))

        return boxes[pick].astype("int")

    def preprocess(self, frame, edging=True, gray=True, blurSize=(5, 5)):
        process_img = frame
        if gray:
            process_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process_img = cv2.GaussianBlur(process_img, blurSize, 0)
        if edging:
            v = np.median(process_img)
            lower = int(max(0, 0.4 * v))
            upper = int(min(255, 1.5 * v))
            process_img = cv2.Canny(process_img, lower, upper)

        return process_img
    
    def extract_mask_color(self, frame, shape=CIRCLE_SHAPE, kernelSize=None):
        frame = self.preprocess(frame, gray=False, edging=False, blurSize=(5, 5))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b, g, r = cv2.split(frame)
        frame = cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

        red_mask = get_red_mask(hsv)
        yellow_mask = get_yellow_mask(hsv)
        blue_mask = get_blue_mask(hsv)

        if shape == CIRCLE_SHAPE:
            mask = cv2.bitwise_or(red_mask, blue_mask)
        elif shape == TRIANGLE_SHAPE:
            mask = cv2.bitwise_or(red_mask, yellow_mask) 
        elif shape == RECTANGLE_SHAPE:
            mask = blue_mask 
        else:
            mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, yellow_mask), blue_mask)

        if kernelSize is not None:
            if shape == CIRCLE_SHAPE:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
            elif shape == RECTANGLE_SHAPE:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
            elif shape == TRIANGLE_SHAPE:
                kernel = np.array([
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ], dtype=np.uint8)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result, mask

    def classify_circle_sign_by_color_ratio(self, roi_img, x_ratio, y_ratio):
        roi_img = prepare_roi_for_matching(roi_img)
        mask = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype=np.uint8)
        center = (roi_img.shape[1] // 2, roi_img.shape[0] // 2)
        radius = min(roi_img.shape[0], roi_img.shape[1]) // 2
        cv2.circle(mask, center, radius, 255, -1)
        masked_img = cv2.bitwise_and(roi_img, roi_img, mask=mask)
        
        hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.bitwise_and(get_red_mask(hsv), mask)
        blue_mask = cv2.bitwise_and(cv2.inRange(hsv, np.array((100, 50, 50)), np.array(BLUE_UPPER)), mask)
        white_mask = cv2.bitwise_and(get_white_mask(hsv), mask)
        yellow_mask = cv2.bitwise_and(get_yellow_mask(hsv), mask)
        black_mask = cv2.bitwise_and(get_black_mask(roi_img), mask)

        total_pixels = cv2.countNonZero(mask)
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        white_pixels = cv2.countNonZero(white_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        black_pixels = cv2.countNonZero(black_mask)
        
        red_ratio = round(red_pixels / total_pixels, 2)
        blue_ratio = round(blue_pixels / total_pixels, 2)
        white_ratio = round(white_pixels / total_pixels, 2)
        yellow_ratio = round(yellow_pixels / total_pixels, 2)
        black_ratio = round(black_pixels / total_pixels, 2)
          
        print(f'{self.frame_count} = Ratios - {total_pixels}: w: {white_pixels} - {white_ratio:.2f} - blk: {black_pixels} - {black_ratio:.2f} - blue: {blue_pixels} - {blue_ratio:.2f} - red: {red_pixels} - {red_ratio:.2f} = x_ratio {x_ratio} + y_ratio {y_ratio}')
          
        if red_ratio >= 0.7 and white_ratio >= 0.03 and (0.4 <= x_ratio < 0.55 or 0.85 <= x_ratio <= 0.9) and 0.09 <= y_ratio <= 0.27:
            return f"Cam Di Nguoc Chieu"
        elif black_ratio >= 0.11 and white_ratio >= 0.05 and yellow_ratio < 0.05 and red_ratio >= 0.13 and (0.38 <= x_ratio < 0.5 or 0.7 <= x_ratio <= 0.9) and (0.37 <= y_ratio <= 0.4 or (0.15 <= y_ratio < 0.2 and not 0.7 <= x_ratio <= 0.9)):
            return f"Cam Re Trai"
        elif (0.12 <= red_ratio <= 0.35 and 0.33 <= blue_ratio <= 0.55) or (0.85 <= x_ratio < 0.9 and 0.2 <= red_ratio <= 0.25 and 0.07 <= blue_ratio <= 0.12):
            return f"Cam Do Xe"
        elif 0.12 <= red_ratio <= 0.4 and 0.15 <= blue_ratio < 0.34:
            return f"Cam Dung Va Do Xe"
        elif blue_ratio >= 0.7 and 0.35 <= x_ratio <= 0.5:
            return f"Huong Vong Chuong Ngai Vat Sang Phai"
        
        return None
 
    def classify_circle_sign(self, roi_bgr, roi_gray, x_global, y_global, width, height):
        label = None
        sift_result = None
        color_result = None
        
        x_ratio = round(x_global / width, 2)
        y_ratio = round(y_global / height, 2)
        
        color_result = self.classify_circle_sign_by_color_ratio(roi_bgr, x_ratio, y_ratio)
        label = color_result
        sift_result = None
        if label is None:
            sift_result = self.feature_match_sift(roi_gray, CIRCLE_SHAPE, min_matches=35)
            if sift_result is not None:
                label = sift_result['name']
        print(label)
        
        return label, sift_result, color_result

    # OK
    def classify_triangle_sign_by_color_ratio(self, roi_img, cnt):
        mask = np.zeros(roi_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        red_mask = cv2.bitwise_and(get_red_mask(hsv), mask)
        yellow_mask = cv2.bitwise_and(get_yellow_mask(hsv), mask)

        total_pixels = cv2.countNonZero(mask)
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        if total_pixels == 0:
            return None

        red_ratio = red_pixels / total_pixels
        yellow_ratio = yellow_pixels / total_pixels

        if red_ratio >= 0.2 and yellow_ratio > 0.4:
            return True

        return False

    def detect_signs(self, frame):
        height, width, _ = frame.shape
        top_crop = 0
        bottom_crop = int(height * 0.5)
        left_crop  = int(width * 0.35)
        cropped_frame = frame[top_crop:bottom_crop, left_crop:]
        
        # Detect Circle
        roi_circle, _ = self.extract_mask_color(cropped_frame, shape=CIRCLE_SHAPE, kernelSize=(5, 5))
        circle_img = self.preprocess(roi_circle, blurSize=(5, 5))
        circles = cv2.HoughCircles(
            circle_img,
            cv2.HOUGH_GRADIENT,
            dp=CIRCLE_DP,
            maxRadius=CIRCLE_MAX_RADIUS,
            minRadius=CIRCLE_MIN_RADIUS,
            minDist=CIRCLE_MIN_DIST,
            param1=CIRCLE_PARAM_1,
            param2=CIRCLE_PARAM_2
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            circle_boxes = []
            for c in circles:
                x, y, r = map(int, c)
                box = [x - r, y - r, 2 * r, 2 * r]
                circle_boxes.append(box)

            final_circle_boxes = self.non_max_suppression(circle_boxes)
            for (x, y, w, h) in final_circle_boxes:
                x_global = x + left_crop
                y_global = y + top_crop
                global_bbox = (x_global, y_global, w, h)
                
                if self.is_existed(global_bbox):
                    continue
                
                # cv2.rectangle(frame, (x_global, y_global), (x_global + w, y_global + h), (255, 255, 255), 5)
                roi_sign = cropped_frame[y:y + h, x:x + w]
                if roi_sign.shape[0] > 10 and roi_sign.shape[1] > 10:
                    roi_gray = cv2.cvtColor(roi_sign, cv2.COLOR_BGR2GRAY)
                    label, sift_result, _ = self.classify_circle_sign(roi_sign, roi_gray, x_global, y_global, width, height)
                    if label:
                        self.sign_tracker.add_tracker(
                            frame, (x_global, y_global, w, h),
                            label,
                            len(sift_result['good_matches']) if sift_result else 0
                        )
                        self.recent_checked_boxes.append(global_bbox)
    
        # Detect triangle -> OK
        _, triangle_mask = self.extract_mask_color(cropped_frame, shape=TRIANGLE_SHAPE, kernelSize=(3, 3))
        contours_tri, _ = cv2.findContours(triangle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_tri:
            x, y, w, h = cv2.boundingRect(cnt)
            tri_crop = cropped_frame[y:y+h, x:x+w]
            if is_triangle_sign_shape(cnt):
                if w > 10 and h > 10:
                    x_global = x + left_crop
                    y_global = y + top_crop
                    cnt_local = cnt.copy()
                    cnt_local[:, 0, 0] -= x
                    cnt_local[:, 0, 1] -= y
                    
                    is_sign = self.classify_triangle_sign_by_color_ratio(tri_crop, cnt_local)
                    if not is_sign:
                        continue
                    
                    best_match = self.feature_match_sift(tri_crop, shape_type=TRIANGLE_SHAPE, min_matches=33)
                    if best_match is not None:
                        self.sign_tracker.add_tracker(frame, (x_global, y_global, w, h), best_match['name'], len(best_match['good_matches']))

        # Detect rectangle -> OK
        _, rect_mask = self.extract_mask_color(cropped_frame, shape=RECTANGLE_SHAPE, kernelSize=(7, 7))
        contours_rect, _ = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_rect:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            vertice = len(approx)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 1.5 < aspect_ratio < 4 and 4 <= vertice < 10 and area > MIN_AREA and area < MAX_AREA:
                rect_crop = cropped_frame[y:y+h, x:x+w]
                best_match = self.feature_match_sift(rect_crop, shape_type=RECTANGLE_SHAPE, min_matches=40)
                if best_match is not None:
                    x_global = x + left_crop
                    y_global = y + top_crop
                    self.sign_tracker.add_tracker(frame, (x_global, y_global, w, h), best_match['name'], len(best_match['good_matches']))

        frame[top_crop:bottom_crop, left_crop:] = cropped_frame
        tracked_boxes = self.sign_tracker.update(frame)
        for (x, y, w, h, _, label, _) in tracked_boxes:
            label = normalize_text(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 3)
            cv2.putText(frame, label, (x - 30, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
         
    def run_video(self, input_path, output_name=OUTPUT_NAME):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Cannot open video {input_path}")
            return
        
        output_path = create_file(output_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        org_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        org_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (org_width, org_height)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.detect_signs(frame)
            
            out.write(frame)
            self.frame_count += 1
        
        cap.release()
        out.release()
        print(f"Processed {self.frame_count} frames. Output saved to {output_path}")

if __name__ == "__main__":
    tracker = TrafficSignRecognizer()
    tracker.run_video("videos/video1.mp4", "output1.mp4")
    tracker.run_video("videos/video2.mp4", "output2.mp4")
