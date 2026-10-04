#!/usr/bin/env python3
"""
opencv_stack.py — infection detection module
------------------------------------------------
Provides a function:
    detect_infected_plants_from_frame(frame)
→ returns (infected_label_block1, infected_label_block2)
"""

import cv2
import numpy as np

# ---------------- Tunable Parameters ----------------
UNION_ERODE_ITER = 4
UNION_DILATE_ITER = 1
AREA_THRESH = 200
# ----------------------------------------------------


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_warp(image):
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=params)
    if ids is None or len(ids) < 4:
        raise RuntimeError("Could not detect 4 ArUco markers.")

    centers = np.array([np.mean(c[0], axis=0) for c in corners])
    global_centroid = centers.mean(axis=0)

    chosen = []
    for c in corners:
        pts = c[0]
        dists = np.linalg.norm(pts - global_centroid, axis=1)
        chosen.append(pts[np.argmax(dists)])
    chosen = np.array(chosen, dtype=np.float32)
    src_rect = order_points(chosen)

    (tl, tr, br, bl) = src_rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))

    ids_list = ids.flatten().tolist()
    warped_marker_corners = {}
    for i, mid in enumerate(ids_list):
        mcorn = np.array(corners[i][0], dtype=np.float32).reshape(-1, 1, 2)
        trans = cv2.perspectiveTransform(mcorn, M).reshape(-1, 2)
        warped_marker_corners[mid] = trans

    return warped, ids.flatten(), warped_marker_corners


def orient_warped_image(warped, ids, warped_marker_corners):
    min_id = int(np.min(ids))
    if min_id not in warped_marker_corners:
        return warped.copy()

    ref_corners = warped_marker_corners[min_id]
    H, W = warped.shape[:2]
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            test = warped.copy()
            tc = ref_corners.copy()
            h, w = H, W
        elif angle == 90:
            test = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            tc = np.array([[H - 1 - y, x] for x, y in ref_corners])
            h, w = W, H
        elif angle == 180:
            test = cv2.rotate(warped, cv2.ROTATE_180)
            tc = np.array([[W - 1 - x, H - 1 - y] for x, y in ref_corners])
            h, w = H, W
        else:
            test = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tc = np.array([[y, W - 1 - x] for x, y in ref_corners])
            h, w = W, H

        cx, cy = tc.mean(axis=0)
        if cx < w / 2 and cy < h / 2:
            br_idx = np.argmax([(x - cx) + (y - cy) for (x, y) in tc])
            brx, bry = tc[br_idx]
            if brx > cx and bry > cy:
                return test
    return warped.copy()


def remove_border_components(bin_mask):
    h, w = bin_mask.shape[:2]
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = np.zeros_like(bin_mask)
    for i in range(1, nb):
        x, y, ww, hh, area = stats[i]
        if x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h:
            continue
        out[labels == i] = 255
    return out


def detect_trays_in_half(half_img):
    hsv = cv2.cvtColor(half_img, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 110, 255]))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 2)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), 2)
    return remove_border_components(mask_white)


def build_mask1_from_trays(half_img, boxes):
    H, W = half_img.shape[:2]
    mask1 = np.zeros((H, W), dtype=np.uint8)
    gray = cv2.cvtColor(half_img, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in boxes:
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        _, roi_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask1[y:y + h, x:x + w] = cv2.bitwise_or(mask1[y:y + h, x:x + w], roi_mask)
    return remove_border_components(mask1)


def build_mask2_union(half_img):
    hsv = cv2.cvtColor(half_img, cv2.COLOR_BGR2HSV)
    mask_g = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    mask_y = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([40, 255, 255]))
    mask2 = cv2.bitwise_or(mask_g, mask_y)
    mask2 = cv2.erode(mask2, np.ones((3, 3), np.uint8), iterations=UNION_ERODE_ITER)
    mask2 = cv2.dilate(mask2, np.ones((3, 3), np.uint8), iterations=UNION_DILATE_ITER)
    return remove_border_components(mask2)


def intersection_masks(mask1, mask2):
    return remove_border_components(cv2.bitwise_and(mask1, mask2))


def two_column_assignment(xs):
    xs = np.array(xs, dtype=float)
    if xs.size == 0:
        return np.array([]), np.array([])
    c1, c2 = xs.min(), xs.max()
    for _ in range(30):
        d1, d2 = np.abs(xs - c1), np.abs(xs - c2)
        labs = (d2 < d1).astype(int)
        if np.any(labs == 0):
            c1 = xs[labs == 0].mean()
        if np.any(labs == 1):
            c2 = xs[labs == 1].mean()
    d1, d2 = np.abs(xs - c1), np.abs(xs - c2)
    return np.array([c1, c2]), (d2 < d1).astype(int)


def find_and_label_blobs(inter_mask, half_img, prefix="P1"):
    contours, _ = cv2.findContours(inter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < AREA_THRESH:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        blobs.append({'box': (x, y, w, h), 'center': (cx, cy), 'area': area})
    if len(blobs) > 6:
        blobs = sorted(blobs, key=lambda b: b['area'], reverse=True)[:6]

    centers_x = [b['center'][0] for b in blobs]
    if not centers_x:
        return [], None
    _, labels = two_column_assignment(centers_x)
    col0 = [blobs[i] for i in range(len(blobs)) if labels[i] == 0]
    col1 = [blobs[i] for i in range(len(blobs)) if labels[i] == 1]
    col0 = sorted(col0, key=lambda b: b['center'][1])
    col1 = sorted(col1, key=lambda b: b['center'][1])
    order = col0 + col1

    letters = list("ABCDEF")
    labeled = []
    for idx, b in enumerate(order):
        x, y, w, h = b['box']
        crop = half_img[y:y + h, x:x + w]
        if crop.size == 0:
            infected_ratio = 0.0
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask_y = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([40, 255, 255]))
            mask_b = cv2.inRange(hsv, np.array([5, 100, 50]), np.array([25, 255, 200]))
            mask_infected = cv2.bitwise_or(mask_y, mask_b)
            infected_ratio = float(np.sum(mask_infected > 0) / (crop.shape[0] * crop.shape[1] + 1e-9))
        label = f"{prefix}{letters[idx]}" if idx < 6 else f"{prefix}{letters[idx % 6]}"
        labeled.append({'full_label': label, 'infected_ratio': infected_ratio})

    best = max(labeled, key=lambda p: p['infected_ratio']) if labeled else None
    return labeled, best


def detect_infected_plants_from_frame(img):
    """
    Main callable function for pico_client.py
    """
    try:
        warped, detected_ids, warped_marker_corners = detect_and_warp(img)
        warped_oriented = orient_warped_image(warped, detected_ids, warped_marker_corners)
        H, W = warped_oriented.shape[:2]
        lower_half = warped_oriented[H // 2:, :]
        left_half = lower_half[:, :W // 2]
        right_half = lower_half[:, W // 2:]

        def process_half(half_img, prefix):
            mask_white = detect_trays_in_half(half_img)
            nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask_white, connectivity=8)
            boxes = [(x, y, w, h) for i, (x, y, w, h, a) in enumerate(stats) if i > 0 and a > 500]
            mask1 = build_mask1_from_trays(half_img, boxes)
            mask2 = build_mask2_union(half_img)
            inter = intersection_masks(mask1, mask2)
            labeled, best = find_and_label_blobs(inter, half_img, prefix)
            return labeled, best

        labeled_left, best_left = process_half(left_half, "P1")
        labeled_right, best_right = process_half(right_half, "P2")
        infected_label_block1 = best_left['full_label'] if best_left else "P1NA"
        infected_label_block2 = best_right['full_label'] if best_right else "P2NA"
        return infected_label_block1, infected_label_block2

    except Exception:
        return "P1A", "P2A"
