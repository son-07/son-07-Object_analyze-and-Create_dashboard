import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# IoU 계산 함수
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Homography 기반 이미지 정렬
def align_images_by_homography(img_ref, img_cur):
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return img_cur, None

    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return img_cur, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return img_cur, None

    height, width = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_cur, H, (width, height))
    return aligned, H

# Homography를 활용해 원본 박스로 복원
def warp_box_back(box, H_inv):
    pts = np.array([
        [box[0], box[1], 1],
        [box[2], box[3], 1]
    ]).T  # (3,2)
    warped = H_inv @ pts
    warped /= warped[2:3, :]
    x1, y1 = warped[0, 0], warped[1, 0]
    x2, y2 = warped[0, 1], warped[1, 1]
    return np.array([int(x1), int(y1), int(x2), int(y2)])

# 이미지 비교 분석 (변화된 부분에만 박스 표시)
def analyze_image_pair(ref_img, img, boxes_current, boxes_ref, H_inv, iou_threshold=0.5, pose_threshold=40):
    abnormal = False
    vis_img = img.copy()
    matched = [False] * len(boxes_current)

    orb = cv2.ORB_create(nfeatures=300)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for ref_cls, ref_box in boxes_ref:
        ref_box = ref_box.astype(np.int32)
        found_match = False
        for i, (cur_cls, cur_box) in enumerate(boxes_current):
            cur_box = cur_box.astype(np.int32)
            if int(ref_cls) != int(cur_cls):
                continue
            score = iou(ref_box, cur_box)
            if score > iou_threshold:
                matched[i] = True
                found_match = True

                rx1, ry1, rx2, ry2 = ref_box
                cx1, cy1, cx2, cy2 = cur_box

                rx1_adj = min(rx2, rx1 + 10)
                rx2_adj = max(rx1_adj, rx2 - 10)
                ry1_adj = min(ry2, ry1 + 10)

                cx1_adj = min(cx2, cx1 + 10)
                cx2_adj = max(cx1_adj, cx2 - 10)
                cy1_adj = min(cy2, cy1 + 10)

                roi_ref = ref_img[ry1_adj:ry2, rx1_adj:rx2_adj]
                roi_cur = img[cy1_adj:cy2, cx1_adj:cx2_adj]

                if roi_ref.size == 0 or roi_cur.size == 0:
                    abnormal = True
                    box_vis = warp_box_back(cur_box, H_inv)
                    cv2.rectangle(vis_img, (box_vis[0], box_vis[1]), (box_vis[2], box_vis[3]), (0, 0, 255), 3)
                    break

                gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
                gray_cur = cv2.cvtColor(roi_cur, cv2.COLOR_BGR2GRAY)

                kp1, des1 = orb.detectAndCompute(gray_ref, None)
                kp2, des2 = orb.detectAndCompute(gray_cur, None)

                if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
                    abnormal = True
                    box_vis = warp_box_back(cur_box, H_inv)
                    cv2.rectangle(vis_img, (box_vis[0], box_vis[1]), (box_vis[2], box_vis[3]), (0, 0, 255), 3)
                    break

                matches = bf.match(des1, des2)
                avg_dist = np.mean([m.distance for m in matches])

                if avg_dist > pose_threshold:
                    abnormal = True
                    box_vis = warp_box_back(cur_box, H_inv)
                    for m in matches:
                        pt = tuple(np.add(np.int32(kp2[m.trainIdx].pt), (box_vis[0], box_vis[1])))
                        cv2.circle(vis_img, pt, 3, (0, 0, 255), -1)
                    cv2.rectangle(vis_img, (box_vis[0], box_vis[1]), (box_vis[2], box_vis[3]), (0, 0, 255), 2)
                break

        if not found_match:
            abnormal = True
            box_vis = warp_box_back(ref_box, H_inv)
            cv2.rectangle(vis_img, (box_vis[0], box_vis[1]), (box_vis[2], box_vis[3]), (0, 0, 255), 2)

    for i, (cls, cur_box) in enumerate(boxes_current):
        if not matched[i]:
            abnormal = True
            cur_box = cur_box.astype(np.int32)
            box_vis = warp_box_back(cur_box, H_inv)
            cv2.rectangle(vis_img, (box_vis[0], box_vis[1]), (box_vis[2], box_vis[3]), (0, 0, 255), 2)

    return vis_img, abnormal

# 시계 전용 SSIM 비교 방식
def analyze_clock_ssim(ref_img, cur_img, box):
    x1, y1, x2, y2 = box.astype(np.int32)
    y1 = min(y2, y1 + 10)
    x1 = min(x2, x1 + 10)
    x2 = max(x1, x2 - 10)
    roi_ref = ref_img[y1:y2, x1:x2]
    roi_cur = cur_img[y1:y2, x1:x2]

    if roi_ref.size == 0 or roi_cur.size == 0:
        return 1.0  # 유사하게 간주

    gray_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(roi_cur, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.resize(gray_ref, (128, 128))
    gray_cur = cv2.resize(gray_cur, (128, 128))

    score, _ = ssim(gray_ref, gray_cur, full=True)
    return score
