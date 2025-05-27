import os
import numpy as np
import cv2
import csv
import re

from constants import *

# File
def create_file(filename, folder=OUTPUT_DIR):
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, filename)
    print(f"File located at: {file_path}")

    return file_path

# Shapes
def angle_between(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    try:
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
        return np.degrees(angle)
    except:
        return 0

def is_triangle(approx, tolerance=0.15):
    if len(approx) != 3:
        return False

    pts = approx.reshape(3, 2)
    sorted_pts = pts[np.argsort(pts[:, 1])]

    top = sorted_pts[0]
    bottom1, bottom2 = sorted_pts[1:]

    is_up = top[1] < bottom1[1] and top[1] < bottom2[1]

    d1 = np.linalg.norm(pts[0] - pts[1])
    d2 = np.linalg.norm(pts[1] - pts[2])
    d3 = np.linalg.norm(pts[2] - pts[0])
    mean_len = (d1 + d2 + d3) / 3.0
    is_equilateral = all(abs(d - mean_len) / mean_len < tolerance for d in [d1, d2, d3])

    return is_up and is_equilateral

def is_triangle_sign_shape(cnt, frame_shape=None):
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)

    if area < MIN_TRIA_AREA or area > MAX_TRIA_AREA:
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False

    hull = cv2.convexHull(approx)
    if len(hull) == 3 or len(approx) == 3:
        return True

    if len(approx) == 4:
        p1, p2, p3 = approx[0][0], approx[1][0], approx[2][0]
        angles = [
            angle_between(p1, p2, p3),
            angle_between(p2, p3, p1),
            angle_between(p3, p1, p2)
        ]
        if all(30 < a < 120 for a in angles) and abs(sum(angles) - 180) < 15:
            return True

    return False

def detect_circles_by_contour(mask_img, min_area=500, min_circularity=0.6):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or area < min_area:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity >= min_circularity:
            x, y, w, h = cv2.boundingRect(cnt)
            circle_boxes.append((x, y, w, h))

    return circle_boxes

def count_red_slash_lines(lines):
    slash_count = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if 115 <= angle <= 155 or 25 <= angle <= 65:
                slash_count += 1
    return slash_count

# Matching
def prepare_roi_for_matching(roi, min_size=64, enhance=True):
    h, w = roi.shape[:2]
    actual_size = (min_size, min_size)
    
    if h < 10 or w < 10:
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
    
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        actual_size = (int(w * scale), int(h * scale))
        roi = cv2.resize(roi, actual_size, interpolation=cv2.INTER_CUBIC)
        
    if not enhance:
        return roi
    
    return enhanced_resize(roi, actual_size)

def enhanced_resize(image, target_size):
    h, w = image.shape[:2]
    
    if h < 50 or w < 50: 
        denoise_h = 5
        clahe_clip = 1.5
        kernel_value = 7
    else:
        denoise_h = 10
        clahe_clip = 2.0
        kernel_value = 9
    
    denoised = cv2.fastNlMeansDenoisingColored(image, None, denoise_h, denoise_h, 7, 21) if len(image.shape) == 3 else cv2.fastNlMeansDenoising(image, None, denoise_h, 7, 21)
    
    kernel = np.array([[-1,-1,-1], 
                      [-1, kernel_value,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    if len(image.shape) == 3:
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced = clahe.apply(sharpened)
    
    if target_size[0] > image.shape[1] or target_size[1] > image.shape[0]:
        resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    
    return resized

# Test
# def prepare_roi_for_matching(roi, min_size=64, enhance=True):
#     h, w = roi.shape[:2]
#     actual_size = (min_size, min_size)

#     if h < 10 or w < 10:
#         roi = cv2.GaussianBlur(roi, (3, 3), 0)

#     if h < min_size or w < min_size:
#         scale = max(min_size / h, min_size / w)
#         actual_size = (int(w * scale), int(h * scale))
#         roi = cv2.resize(roi, actual_size, interpolation=cv2.INTER_CUBIC)

#     if not enhance:
#         return roi

#     return enhanced_resize(roi, actual_size)

# def enhanced_resize(image, target_size):
#     h, w = image.shape[:2]

#     # Cấu hình theo kích thước ảnh
#     if h < 50 or w < 50:
#         denoise_h = 5
#         clahe_clip = 1.5
#         kernel_value = 7
#     else:
#         denoise_h = 10
#         clahe_clip = 2.0
#         kernel_value = 9

#     # Denoising
#     if len(image.shape) == 3:
#         denoised = cv2.fastNlMeansDenoisingColored(image, None, denoise_h, denoise_h, 7, 21)
#     else:
#         denoised = cv2.fastNlMeansDenoising(image, None, denoise_h, 7, 21)

#     # Sharpen
#     kernel = np.array([[-1, -1, -1],
#                        [-1, kernel_value, -1],
#                        [-1, -1, -1]])
#     sharpened = cv2.filter2D(denoised, -1, kernel)

#     # CLAHE
#     clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
#     if len(image.shape) == 3:
#         lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         l = clahe.apply(l)
#         enhanced = cv2.merge((l, a, b))
#         enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
#     else:
#         enhanced = clahe.apply(sharpened)

#     # Resize chuẩn cuối cùng
#     resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4 if target_size[0] > image.shape[1] else cv2.INTER_AREA)

#     # 👉 Tăng saturation (giúp nhận diện viền đỏ rõ hơn)
#     if len(image.shape) == 3:
#         hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
#         hsv[..., 1] = cv2.equalizeHist(hsv[..., 1])
#         resized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     # 👉 (Tùy chọn) Làm nét thêm lần nữa bằng Unsharp Mask
#     # resized = cv2.addWeighted(resized, 1.5, cv2.GaussianBlur(resized, (0, 0), 3), -0.5, 0)

#     return resized


# Mask
def get_red_mask(hsv):
    return cv2.bitwise_or(
        cv2.inRange(hsv, np.array(RED_LOWER1), np.array(RED_UPPER1)),
        cv2.inRange(hsv, np.array(RED_LOWER2), np.array(RED_UPPER2))
    )

def get_yellow_mask(hsv):
    return cv2.inRange(hsv, np.array(YELLOW_LOWER), np.array(YELLOW_UPPER))

def get_blue_mask(hsv):
    return cv2.inRange(hsv, np.array(BLUE_LOWER), np.array(BLUE_UPPER))

def get_black_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.inRange(gray, 0, 50)

def get_white_mask(hsv):
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    return cv2.inRange(hsv, lower_white, upper_white)

# Log
def log_color_to_csv(frame_count, box, label, csv_path='color_stats.csv'):
    header = ['frame', 'box', 'label']
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'frame': frame_count,
            'box': box,
            'label': label or "None"
        })

def show_hsv_values(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = hsv[y, x]
            print(f"HSV at ({x},{y}): {pixel}")

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouse_callback)

    print("Click on the image to get HSV values.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Line
import numpy as np
import cv2

def group_similar_lines(lines, angle_thresh=10, dist_thresh=30):
    if lines is None:
        return []

    groups = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        added = False
        for group in groups:
            gx1, gy1, gx2, gy2 = group['avg_line']
            g_angle = group['angle']
            g_center = group['center']

            if abs(angle - g_angle) < angle_thresh:
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                dist = np.linalg.norm(np.array(center) - np.array(g_center))

                if dist < dist_thresh:
                    group['lines'].append((x1, y1, x2, y2))
                    group['center'] = tuple(np.mean(group['lines'], axis=0)[0:2])
                    group['avg_line'] = tuple(np.mean(group['lines'], axis=0).astype(int))
                    added = True
                    break

        if not added:
            group = {
                'lines': [(x1, y1, x2, y2)],
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'avg_line': (x1, y1, x2, y2),
                'angle': angle
            }
            groups.append(group)

    return groups

# Label
def normalize_text(filename):
    name = filename.rsplit(".", 1)[0]
    
    if "_" in name:
        name = name.replace("_", " ")
        name = re.sub(r'\d+$', '', name)  # Loại bỏ số ở cuối nếu có
    
    return name.strip()