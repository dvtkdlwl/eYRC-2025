#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np
import json
import os
import time

# ---------------- Tunable Parameters ----------------
UNION_ERODE_ITER = 4
UNION_DILATE_ITER = 1
AREA_THRESH = 200
# ----------------------------------------------------


# ---------------- ArUco Homography (ONCE) ----------------

def compute_homography_from_ids(image):
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

    try:
        params = aruco.DetectorParameters()
    except:
        params = aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=params)

    if ids is None:
        return None, None

    ids = ids.flatten()
    centers = {}

    for c, i in zip(corners, ids):
        centers[i] = np.mean(c[0], axis=0)

    required_ids = {80, 85, 90, 95}
    if not required_ids.issubset(centers.keys()):
        return None, None

    # Fixed orientation:
    # 80 → TL, 85 → TR, 90 → BR, 95 → BL
    src = np.array([
        centers[80],
        centers[85],
        centers[90],
        centers[95],
    ], dtype=np.float32)

    width = int(max(
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[3])
    ))

    height = int(max(
        np.linalg.norm(src[3] - src[0]),
        np.linalg.norm(src[2] - src[1])
    ))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    return H, (width, height)


# ---------------- Vision Utilities ----------------

def remove_border_components(mask):
    h, w = mask.shape
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, nb):
        x, y, ww, hh, _ = stats[i]
        if x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h:
            continue
        out[labels == i] = 255
    return out


def detect_plants(half, solid_trays):
    hsv = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 255), (180, 0, 255))
    gated_plants = cv2.bitwise_and(cv2.bitwise_not(mask), solid_trays)
    cv2.imwrite("/home/divit/pico_ws2/debug_images/gated_plants.png", gated_plants)
    return remove_border_components(mask)


def detect_trays_only(half, prefix):
    hsv = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        (0, 0, 255),
        (0, 7, 255)
    )

    img_h, img_w = mask.shape[:2]
    edge = 5 # pixels

    # --- BLACK OUT EDGES ---
    mask[0:edge, :] = 0                 # top
    mask[img_h-edge:img_h, :] = 0       # bottom
    mask[:, 0:edge] = 0                 # left
    mask[:, img_w-edge:img_w] = 0       # right

    debug_img = np.zeros_like(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        if area < 25000 or area > 45000:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw // 2, y + bh // 2

        valid_contours.append((cx, cy, c))
        #print(f"[{prefix}] Contour {i}: area={area:.1f}, center=({cx},{cy})")

        cv2.rectangle(
            debug_img,
            (x, y),
            (x + bw, y + bh),
            255,
            -1
        )

    #print("Number of tray contours:", len(valid_contours))
    return debug_img


def refine_plant_blobs(mask, debug_dir=None, tag=""):
    """
    Refines plant blobs inside tray using tuned morphology.
    
    Input:
        mask (uint8): binary image (0 or 255) where white = plant pixels
    Output:
        refined (uint8): clean, separated plant blobs
    """

    # --- YOUR FINAL TUNED PARAMETERS ---
    erode_size = 7
    erode_iter = 2
    dilate_size = 7
    dilate_iter = 2
    close_size = 7
    close_iter = 1
    kernel_shape = cv2.MORPH_ELLIPSE
    # ----------------------------------

    # Structuring elements
    erode_kernel = cv2.getStructuringElement(
        kernel_shape, (erode_size, erode_size)
    )
    dilate_kernel = cv2.getStructuringElement(
        kernel_shape, (dilate_size, dilate_size)
    )
    close_kernel = cv2.getStructuringElement(
        kernel_shape, (close_size, close_size)
    )

    # 1️⃣ OPEN → break thin leaf bridges
    opened = cv2.erode(mask, erode_kernel, iterations=erode_iter)
    opened = cv2.dilate(opened, dilate_kernel, iterations=dilate_iter)

    # 2️⃣ CLOSE → fill holes, solidify blobs
    refined = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=close_iter
    )

    # Optional debug
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/{tag}_01_input.png", mask)
        cv2.imwrite(f"{debug_dir}/{tag}_02_opened.png", opened)
        cv2.imwrite(f"{debug_dir}/{tag}_03_refined.png", refined)

    return refined



def find_best_infected(mask, half, prefix, debug_dir=None, frame_id=""):
    """
    mask      : binary image where white blobs = plants (uint8, 0/255)
    half      : corresponding BGR image (left or right half)
    prefix    : "P1" or "P2"
    debug_dir : path to debug folder
    frame_id  : optional string for filenames
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_ratio = 0.0
    best_label = f"{prefix}NA"
    letters = ["A", "B", "C", "D", "E", "F"]

    # ---------- DEBUG IMAGE ----------
    debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    valid_contours = []

    # ---------- FILTER + COLLECT CONTOURS ----------
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        if area < 800 or area > 4000:
            cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 1)
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2

        # Reject long thin junk
        if (w * h > 6000):
            continue

        if (w / h > 2.5) or (h / w > 2.5):
            continue

        valid_contours.append((cx, cy, c))

        #print(f"[{prefix}] Contour {i}: area={area:.1f}, center=({cx},{cy})")

        # draw accepted blob
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # ---------- ALWAYS SAVE DEBUG ----------
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(debug_dir, f"{prefix}_contours_raw_{frame_id}.png"),
            debug_img
        )
        cv2.imwrite(
            os.path.join(debug_dir, f"{prefix}_mask_{frame_id}.png"),
            mask
        )

    # ---------- VALIDATION ----------
    if len(valid_contours) != 6:
        print(f"[{prefix}] Invalid contour count: {len(valid_contours)} (expected 6)")
        return best_label

    # ---------- COLUMN-WISE ORDERING ----------
    xs = [cx for cx, _, _ in valid_contours]
    x_mid = np.median(xs)

    left_col = []
    right_col = []

    for cx, cy, c in valid_contours:
        if cx < x_mid:
            left_col.append((cx, cy, c))
        else:
            right_col.append((cx, cy, c))

    if len(left_col) != 3 or len(right_col) != 3:
        print(f"[{prefix}] Column split failed: L={len(left_col)}, R={len(right_col)}")
        return best_label

    left_col.sort(key=lambda x: x[1])   # top → bottom
    right_col.sort(key=lambda x: x[1])

    ordered_contours = left_col + right_col  # A B C D E F

    # ---------- HSV INFECTION CHECK ----------
    for idx, (_, _, c) in enumerate(ordered_contours):
        x, y, w, h = cv2.boundingRect(c)
        crop = half[y:y + h, x:x + w]

        if crop.size == 0:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # YOUR FINAL TUNED HSV VALUES
        mask_y = cv2.inRange(hsv, (19, 109, 38), (29, 255, 255))
        mask_b = cv2.inRange(hsv, (5, 100, 50), (25, 255, 200))
        infected = cv2.bitwise_or(mask_y, mask_b)

        ratio = np.sum(infected > 0) / (crop.shape[0] * crop.shape[1] + 1e-9)

        #print(f"[{prefix}{letters[idx]}] infected_ratio = {ratio:.3f}")

        if ratio > best_ratio:
            best_ratio = ratio
            best_label = f"{prefix}{letters[idx]}"

            # highlight infected plant
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # HSV debug
        if debug_dir is not None:
            cv2.imwrite(
                os.path.join(debug_dir, f"{prefix}_{letters[idx]}_infected_{frame_id}.png"),
                infected
            )

    # ---------- FINAL DEBUG IMAGE ----------
    if debug_dir is not None:
        cv2.imwrite(
            os.path.join(debug_dir, f"{prefix}_final_{frame_id}.png"),
            debug_img
        )

    return best_label


# ---------------- ROS2 Node ----------------

class Task4ANode(Node):
    def __init__(self):
        super().__init__('task4a_node')

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            '/detected_plants',
            10
        )

        # Lock-on variables
        self.homography = None
        self.warp_size = None
        self.last_output = None

        self.debug_dir = os.path.expanduser('~/pico_ws2/debug_images')
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_saved = False



    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ---------- Lock arena ONCE ----------
        if self.homography is None:
            H, size = compute_homography_from_ids(frame)
            if H is None:
                return

            self.homography = H
            self.warp_size = size
            self.get_logger().info("Arena locked using ArUco IDs")
            return

        # ---------- Reuse homography ----------
        warped = cv2.warpPerspective(
            frame,
            self.homography,
            self.warp_size
        )

        H, W = warped.shape[:2]
        lower = warped[H // 2:, :]
        left = lower[:, :W // 2]
        right = lower[:, W // 2:]

        

        maskonlyl = detect_trays_only(left, "P1")
        mask_l = detect_plants(left, maskonlyl)
        cv2.imwrite("/home/divit/pico_ws2/debug_images/maskonlyl.png", maskonlyl)
        left_block_final_plants_binary = refine_plant_blobs(cv2.bitwise_not(mask_l), debug_dir='/home/divit/pico_ws2/debug_images')
        left_block_final_plants_binary = cv2.bitwise_and(left_block_final_plants_binary, maskonlyl)
        left_block_final_plants = cv2.bitwise_and(left, left, mask=left_block_final_plants_binary)
        cv2.imwrite("/home/divit/pico_ws2/debug_images/final_left_block.png", left_block_final_plants)
        inter_l = cv2.bitwise_and(maskonlyl, cv2.bitwise_not(mask_l))

        maskonlyr = detect_trays_only(right, "P2")
        mask_r = detect_plants(right, maskonlyr)
        cv2.imwrite("/home/divit/pico_ws2/debug_images/maskonlyr.png", maskonlyr)
        right_block_final_plants_binary = refine_plant_blobs(cv2.bitwise_not(mask_r), debug_dir='/home/divit/pico_ws2/debug_images')
        right_block_final_plants_binary = cv2.bitwise_and(right_block_final_plants_binary, maskonlyr)
        right_block_final_plants = cv2.bitwise_and(right, right, mask=right_block_final_plants_binary)
        cv2.imwrite("/home/divit/pico_ws2/debug_images/final_right_block.png", right_block_final_plants)
        inter_r = cv2.bitwise_and(maskonlyr, cv2.bitwise_not(mask_r))


        # ---------- DEBUG: save cropped halves ONCE ----------
        if not self.debug_saved:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            cv2.imwrite(
                os.path.join(self.debug_dir, f"warped_{timestamp}.png"),
                warped
            )

            cv2.imwrite(
                os.path.join(self.debug_dir, f"left_block_{timestamp}.png"),
                left
            )

            cv2.imwrite(
                os.path.join(self.debug_dir, f"right_block_{timestamp}.png"),
                right
            )

            cv2.imwrite(
                os.path.join(self.debug_dir, f"inter_left_{timestamp}.png"),
                inter_l
            )

            cv2.imwrite(
                os.path.join(self.debug_dir, f"inter_right_{timestamp}.png"),
                inter_r
            )

            cv2.imwrite(
                os.path.join(self.debug_dir, "mask_l.png"),
                mask_l
            )

            cv2.imwrite(
                os.path.join(self.debug_dir, "mask_r.png"),
                mask_r
            )
            

            self.get_logger().info("Saved warped, left, and right debug images")
            self.debug_saved = True

        p1 = find_best_infected(left_block_final_plants_binary, left, "P1")
        p2 = find_best_infected(right_block_final_plants_binary, right, "P2")

        result = [p1, p2]

        if result != self.last_output:
            msg_out = String()
            msg_out.data = json.dumps(result)
            self.pub.publish(msg_out)
            self.last_output = result
            self.get_logger().info(f"Detected: {result}")


def main(args=None):
    rclpy.init(args=args)
    node = Task4ANode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()