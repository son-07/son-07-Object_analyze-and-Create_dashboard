import os
import cv2
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
import requests
from analyze_dashboard_pkg.img_utils import (
    align_images_by_homography,
    analyze_image_pair,
    analyze_clock_ssim
)
import numpy as np

class ImageAnalyzerNode(Node):
    def __init__(self):
        super().__init__('image_analyzer_node')
        self.base_dir = '/home/son/sw_project_2025/ros2_ws/src/analyze_dashboard_pkg/object'
        self.upload_dir = '/home/son/sw_project_2025/ros2_ws/src/analyze_dashboard_pkg/web_dashboard/uploads'
        os.makedirs(self.upload_dir, exist_ok=True)
        self.yolo = YOLO('yolov8n.pt')
        self.analyzed = set()
        self.timer = self.create_timer(5.0, self.analyze_all_folders)

    def analyze_all_folders(self):
        for i in range(10):
            folder_name = f'object{i}'
            folder_path = os.path.join(self.base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            if len(files) < 2:
                continue

            base_img_path = os.path.join(folder_path, files[0])
            base_img = cv2.imread(base_img_path)
            base_results = self.yolo(base_img, verbose=False)[0]
            base_boxes = list(zip(base_results.boxes.cls.cpu().numpy(),
                                  base_results.boxes.xyxy.cpu().numpy()))

            for fname in files[1:]:
                if (folder_name, fname) in self.analyzed:
                    continue

                img_path = os.path.join(folder_path, fname)
                img = cv2.imread(img_path)
                aligned_img, H = align_images_by_homography(base_img, img)

                if H is None:
                    self.get_logger().warn(f'{fname}: Homography 실패')
                    continue

                results = self.yolo(aligned_img, verbose=False)[0]
                cur_boxes = list(zip(results.boxes.cls.cpu().numpy(),
                                     results.boxes.xyxy.cpu().numpy()))

                if folder_name == 'object4':  # 시계 전용
                    abnormal = False
                    vis_img = img.copy()
                    for _, box in base_boxes:
                        score = analyze_clock_ssim(base_img, aligned_img, box)
                        if score < 0.92:
                            abnormal = True
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    abnormal_img = vis_img
                else:
                    H_inv = np.linalg.inv(H)
                    abnormal_img, abnormal = analyze_image_pair(base_img, img, cur_boxes, base_boxes, H_inv)

                if abnormal:
                    upload_path = os.path.join(self.upload_dir, f'{folder_name}_analyzed_{fname}')
                    cv2.imwrite(upload_path, abnormal_img)
                    self.send_to_dashboard(upload_path, tag='이상 감지')
                    self.get_logger().info(f'{folder_name}/{fname}: 이상 감지됨. 결과 전송 완료.')
                else:
                    self.get_logger().info(f'{folder_name}/{fname}: 이상 없음.')

                self.analyzed.add((folder_name, fname))

        self.check_failed_folder()

    def check_failed_folder(self):
        failed_path = os.path.join(self.base_dir, 'failed')
        if not os.path.isdir(failed_path):
            return

        files = sorted([
            f for f in os.listdir(failed_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        for fname in files:
            if ('failed', fname) in self.analyzed:
                continue

            img_path = os.path.join(failed_path, fname)
            upload_path = os.path.join(self.upload_dir, f'failed_{fname}')
            img = cv2.imread(img_path)
            cv2.imwrite(upload_path, img)
            self.send_to_dashboard(upload_path, tag='촬영 실패')
            self.get_logger().info(f'failed/{fname}: 촬영 실패. 결과 전송 완료.')
            self.analyzed.add(('failed', fname))

    def send_to_dashboard(self, img_path, tag='이상 감지'):
        try:
            with open(img_path, 'rb') as f:
                files = {'image': (os.path.basename(img_path), f, 'image/jpeg')}
                data = {
                    'filename': os.path.basename(img_path),
                    'matches': tag
                }
                res = requests.post('http://localhost:8000/upload_result/', files=files, data=data, timeout=3)
                self.get_logger().info(f'대시보드 응답: {res.status_code}')
        except Exception as e:
            self.get_logger().warn(f'전송 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageAnalyzerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
