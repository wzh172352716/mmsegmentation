from typing import Optional, Sequence, Dict
import json
import numpy as np
import cv2
import os
import os.path as osp
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
import shutil
from mmengine.logging import MMLogger


def mkdir_or_exist(dir_name):
    """如果目录不存在，则创建该目录"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

@METRICS.register_module()
class MissedPedestrianMetric(BaseMetric):
    def __init__(self,
             pedestrian_idx: int = 11,
             lane_idx: int = 0,
             sidewalk_idx: int = 1,
             collect_device: str = 'cpu',
             prefix: Optional[str] = None,
             fx: float = 2262.52,  # 水平方向焦距
             fy: float = 2265.30,  # 垂直方向焦距
             cx: float = 1096.98,  # 图像中心 x 坐标
             cy: float = 513.137,  # 图像中心 y 坐标
             image_width: int = 2048,
             image_height: int = 1024,
             camera_height: float = 0.37,  # 相机高度（米）
             pitch_angle: float = 0.03,  # 相机俯仰角（弧度）
             known_height: float = 1.7,  # 行人的实际高度
             min_area_threshold: int = 100,
             output_dir: Optional[str] = './pedestrians_detection',
             min_width: int = 10,
             min_height: int = 10,
             max_images_to_save: int = 200,
             iou_threshold_min: float = 0,
             iou_threshold_max: float = 0.5,
             **kwargs) -> None:
    
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        # 通用参数
        self.pedestrian_idx = pedestrian_idx
        self.lane_idx = lane_idx
        self.sidewalk_idx = sidewalk_idx
        self.collect_device = collect_device
        self.prefix = prefix
        
        # 相机参数
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height
        self.camera_height = camera_height
        self.pitch_angle = pitch_angle
        self.known_height = known_height

        # 检测参数
        self.min_area_threshold = min_area_threshold
        self.iou_threshold_min = iou_threshold_min
        self.iou_threshold_max = iou_threshold_max
        self.min_width = min_width
        self.min_height = min_height

        # 输出目录
        self.output_dir = osp.abspath(output_dir) if output_dir else None
        self.output_visual_dir = osp.join(self.output_dir, 'missed') if self.output_dir else None
        self.output_detected_dir = osp.join(self.output_dir, 'detected') if self.output_dir else None

        if self.output_dir:
            mkdir_or_exist(self.output_dir)
            mkdir_or_exist(self.output_visual_dir)
            mkdir_or_exist(self.output_detected_dir)

        # 保存文件相关
        self.saved_image_paths = []
        self.max_images_to_save = max_images_to_save

        # 数据存储
        self.results = []
        self.image_info = []  # 用于存储每张图像的相关信息
        self.equidistant_lines = []  # 每个元素是 (像素高度, 距离)
        self.missed_pedestrian_dict = {}
        self.detected_pedestrians_dict = {}


    def draw_ground_lines(self, image, num_lines: int = 15, pedestrian_boxes: list = None) -> None:
        """
        绘制等距线并标注行人边界框与距离。
        """
        height, width = image.shape[:2]

        # 相机内参
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        h = self.camera_height
        theta_pitch = self.pitch_angle

        # 清空等距线数据
        self.equidistant_lines.clear()

        # 设置实际距离范围
        max_distance = 25  # 最大距离（米）
        min_distance = 1   # 最小距离（米）
        distances = np.arange(min_distance, max_distance + 1, 1)
        if num_lines is not None:
            distances = distances[:num_lines]

        for d in distances:
            u_list = []
            v_list = []

            # 角度范围，从 -90 到 90 度
            for angle_deg in range(-90, 91, 1):
                angle_rad = np.deg2rad(angle_deg)
                X_world = d * np.sin(angle_rad)
                Z_world = d * np.cos(angle_rad)

                # 相机坐标系下的点
                X_cam = X_world
                Y_cam = h
                Z_cam = Z_world

                # 旋转矩阵
                R_pitch = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta_pitch), -np.sin(theta_pitch)],
                    [0, np.sin(theta_pitch), np.cos(theta_pitch)]
                ])

                point_cam = np.array([X_cam, Y_cam, Z_cam])
                point_cam_rotated = R_pitch @ point_cam

                X_c = point_cam_rotated[0]
                Y_c = point_cam_rotated[1]
                Z_c = point_cam_rotated[2]

                if Z_c <= 0:
                    continue

                # 投影到图像坐标系
                u = fx * X_c / Z_c + cx
                v = fy * Y_c / Z_c + cy

                # 检查点是否在图像范围内
                if 0 <= u < width and 0 <= v < height:
                    u_list.append(int(u))
                    v_list.append(int(v))

            # 如果有足够的点，绘制等距线
            if len(u_list) > 1:
                avg_v = int(np.mean(v_list))
                self.equidistant_lines.append((avg_v, d))

                # 绘制等距线
                pts = np.array([[u, v] for u, v in zip(u_list, v_list)], dtype=np.int32)
                cv2.polylines(image, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                mid_index = len(u_list) // 2
                cv2.putText(image, f"{d:.1f}m", (u_list[mid_index], v_list[mid_index] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 如果传入了行人边界框，标注每个行人的距离
        if pedestrian_boxes:
            for box in pedestrian_boxes:
                x, y, w, h = box['bounding_box']
                lowest_point = y + h  # 获取边界框的最低点
                distance = self.get_distance_from_lines(lowest_point, self.equidistant_lines)

                # 在图像上绘制距离信息
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制边界框
                cv2.putText(image, f'{distance:.1f}m', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                

    def get_distance_from_lines(self, pixel_y, equidistant_lines):
        """
        根据像素 y 坐标，找到对应的等距线，返回对应的距离。
        """
        # 按照像素高度从小到大排序（即从图像顶部到底部）
        equidistant_lines_sorted = sorted(equidistant_lines, key=lambda x: x[0])

        # 遍历等距线，找到最低像素点所在的区间
        for idx, (line_y, distance) in enumerate(equidistant_lines_sorted):
            if pixel_y <= line_y:
                # 行人在该等距线之上，返回对应的距离
                return distance
        # 如果低于所有等距线，返回最大距离
        return equidistant_lines_sorted[-1][1]


    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:

        logger = MMLogger.get_current_instance()

        for data_sample in data_samples:
            # 获取图像的基础名称和路径
            img_path = data_sample['img_path']
            basename = osp.splitext(osp.basename(img_path))[0]

            # 提取预测和真实标签的行人区域
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()

            label_pedestrian = self.extract_mask(label, self.pedestrian_idx)
            pred_label_pedestrian = self.extract_mask(pred_label, self.pedestrian_idx)

            # 提取预测的车道和人行道区域
            pred_label_lane = self.extract_mask(pred_label, self.lane_idx)
            pred_label_sidewalk = self.extract_mask(pred_label, self.sidewalk_idx)

            # 读取原始图像
            original_img = cv2.imread(img_path)

            # 查找漏检的行人和检测到的行人，并在图像上标记
            missed_pedestrians, detected_pedestrians = self.visualize_missed_pedestrians(
                original_img, label_pedestrian, pred_label_pedestrian, pred_label_lane, pred_label_sidewalk, pred_label)

            # 提取漏检和检测到的行人边界框
            all_pedestrian_boxes = []
            for ped in detected_pedestrians:
                all_pedestrian_boxes.append({'bounding_box': ped['bounding_box'], 'type': 'detected'})
            for ped in missed_pedestrians:
                all_pedestrian_boxes.append({'bounding_box': ped['bounding_box'], 'type': 'missed'})

            # 在图像上绘制等距线
            self.draw_ground_lines(original_img, num_lines=15)

            # 标注每个行人的距离
            for box in all_pedestrian_boxes:
                x, y, w, h = box['bounding_box']
                lowest_point = y + h  # 行人边界框的最低点像素
                distance = self.get_distance_from_lines(lowest_point, self.equidistant_lines)

                # 在框的左边标注距离
                label_color = (255, 0, 0) if box['type'] == 'missed' else (0, 255, 255)  # 红色标注漏检行人，黄色标注检测到的行人
                cv2.putText(original_img, f'{distance:.1f}m', (x - 60, y + h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

            # 保存检测到的行人信息
            if detected_pedestrians:
                self.detected_pedestrians_dict[basename] = detected_pedestrians

                # 保存检测到的行人可视化图像
                if self.output_dir is not None and (len(self.saved_image_paths) < self.max_images_to_save):
                    detected_save_path = osp.join(self.output_detected_dir, f"{basename}_detected.jpg")
                    cv2.imwrite(detected_save_path, original_img)
                    self.saved_image_paths.append(detected_save_path)

            # 保存漏检行人信息
            if missed_pedestrians:
                self.missed_pedestrian_dict[basename] = missed_pedestrians

                # 保存漏检行人的可视化图像
                if self.output_dir is not None and len(self.saved_image_paths) < self.max_images_to_save:
                    missed_save_path = osp.join(self.output_visual_dir, f"{basename}_missed.jpg")
                    cv2.imwrite(missed_save_path, original_img)
                    self.saved_image_paths.append(missed_save_path)

        # 将漏检行人字典保存到 JSON 文件
        missed_output_file_path = osp.join(self.output_dir, "missed_pedestrians.json")
        with open(missed_output_file_path, 'w') as f:
            json.dump(self.missed_pedestrian_dict, f, indent=4)

        # 将检测到的行人字典保存到 JSON 文件
        detected_output_file_path = osp.join(self.output_dir, "detected_pedestrians.json")
        with open(detected_output_file_path, 'w') as f:
            json.dump(self.detected_pedestrians_dict, f, indent=4)






    def save_image(self, visual_img, basename, img_path):
        """保存原始和带标注的图像"""
        mkdir_or_exist(self.output_orig_dir)
        mkdir_or_exist(self.output_visual_dir)

        # 保存原始图像
        orig_save_path = osp.join(self.output_orig_dir, f"{basename}_original.png")
        shutil.copy(img_path, orig_save_path)

        # 保存标注后的图像
        visual_save_path = osp.join(self.output_visual_dir, f"{basename}_missed_visualized.png")
        cv2.imwrite(visual_save_path, visual_img)  # 确保 visual_img 是修改后的图像

        self.saved_image_paths.append((orig_save_path, visual_save_path))
        

    def extract_mask(self, label, idx):
        """提取特定索引的二值掩码"""
        return np.where(label == idx, 255, 0).astype("uint8")
    

    def compute_iou_contours(self, contour1, contour2, mask_shape):
        """基于轮廓计算 IoU"""
        mask1 = np.zeros(mask_shape, dtype=np.uint8)
        mask2 = np.zeros(mask_shape, dtype=np.uint8)
        
        # 绘制轮廓
        cv2.drawContours(mask1, [contour1], -1, color=1, thickness=cv2.FILLED)
        cv2.drawContours(mask2, [contour2], -1, color=1, thickness=cv2.FILLED)
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union != 0 else 0


    def find_contours(self, mask):
        """查找轮廓"""
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


    def visualize_missed_pedestrians(self, img, label_pedestrian, pred_label_pedestrian, pred_label_lane, pred_label_sidewalk, pred_label):
        """在图像上标出漏检的行人区域，仅在满足 IoU 在 0 到 0.3 之间时绘制红框和轮廓"""

        missed_pedestrians = [] # 用于存储 IoU < 0.3 的行人信息
        detected_pedestrians = []  # 用于存储 IoU > 0.3 的行人信息

        img[pred_label_lane == 255] = (230, 216, 173)  # 淡蓝色填充预测车道区域
        img[pred_label_sidewalk == 255] = (250, 230, 230)   # 浅紫色填充预测人行道区域

        # 获取真实标签和预测标签的行人轮廓
        gt_contours = self.find_contours(label_pedestrian)
        pred_contours = self.find_contours(pred_label_pedestrian)


                # 检查每个真实行人轮廓是否为漏检
        for gt_contour in gt_contours:
            x, y, w, h = cv2.boundingRect(gt_contour)

            # 过滤掉较小的行人区域
            if w < self.min_width or h < self.min_height:
                continue

#-------------------------------------------------------------------------------------------

        #------------------------------------------------------------------------

            # 初始化最大 IoU 为 0
            max_iou = 0
            best_pred_contour = None
            for pred_contour in pred_contours: 
                # 计算 GT 和 Pred 轮廓之间的 IoU
                iou = self.compute_iou_contours(gt_contour, pred_contour, label_pedestrian.shape)
                if iou > max_iou:
                    max_iou = iou
                    best_pred_contour = pred_contour  # 保存 IoU 最大的预测轮廓


                    # 初始化最大 IoU 为 0
                    max_iou = 0
                    best_pred_contour = None
                    for pred_contour in pred_contours: 
                        # 计算 GT 和 Pred 轮廓之间的 IoU
                        iou = self.compute_iou_contours(gt_contour, pred_contour, label_pedestrian.shape)
                        if iou > max_iou:
                            max_iou = iou
                            best_pred_contour = pred_contour  # 保存 IoU 最大的预测轮廓

#------------------------------------------------------- iou>0.3------------------------------------------------------#
            if max_iou > 0.3:
                if best_pred_contour is not None:
                    # 获取预测轮廓的外接矩形
                    x_pred, y_pred, w_pred, h_pred = cv2.boundingRect(best_pred_contour)

                    # 确保数据类型是标准的 Python 类型
                    x_pred, y_pred, w_pred, h_pred = int(x_pred), int(y_pred), int(w_pred), int(h_pred)

                    # 绘制蓝色框
                    cv2.rectangle(img, (x_pred, y_pred), (x_pred + w_pred, y_pred + h_pred), (255, 0, 0), 2)

                    # 提取框中的像素区域
                    rect_pixels = pred_label[y_pred:y_pred + h_pred, x_pred:x_pred + w_pred]

                    # 计算框中车道和人行道的像素面积
                    lane_area = np.sum(rect_pixels == self.lane_idx)  # self.lane_idx 是车道的索引
                    sidewalk_area = np.sum(rect_pixels == self.sidewalk_idx)  # self.sidewalk_idx 是人行道的索引

                    # 比较人行道面积与车道面积的比例
                    location_label = ""
                    if sidewalk_area > 1.5 * lane_area:  # 人行道面积大于车道面积的 1.5 倍
                        location_label = "Sidewalk"
                    elif lane_area > 0:  # 如果车道面积不为 0 且比例较高
                        location_label = "Lane"

                    # 在框右侧显示标签
                    if location_label:
                        text_x = x_pred + w_pred + 5
                        text_y = y_pred + h_pred // 2
                        cv2.putText(img, location_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 显示 IoU 值
                    cv2.putText(img, f"IoU={float(max_iou):.2f}", (x_pred, y_pred - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    # 存储当前行人的信息
                    detected_pedestrians.append({
                        'bounding_box': (x_pred, y_pred, w_pred, h_pred),  # 边界框
                        'iou': float(max_iou),                            # IoU 值
                        'location_label': location_label,                 # 区域标签（Lane/Sidewalk）
                        'lane_area': int(lane_area),                      # 车道面积
                        'sidewalk_area': int(sidewalk_area)               # 人行道面积
                    })



#-------------------------------------------------------------0.02<iou<0,3---------------------------------------------------#
            # 只保留 IoU 在 0.02 到 0.3 之间的行人
            if 0.02 <= max_iou <= 0.3:
                if best_pred_contour is not None:
                    # 获取预测轮廓的外接矩形
                    x_pred, y_pred, w_pred, h_pred = cv2.boundingRect(best_pred_contour)

                    # 计算延伸后的框的底边坐标
                    extended_y_bottom = y_pred + h_pred
                    img_height = img.shape[0]
                    
                    # 向下延伸，直到与车道或人行道区域相交
                    location_label = ""
                    while extended_y_bottom < img_height:
                        # 检查当前延伸后的底边是否与车道或人行道有交集
                        extended_rect = img[extended_y_bottom:extended_y_bottom + 1, x_pred:x_pred + w_pred]
                        lane_intersection = np.any(extended_rect == (230, 216, 173))  # 车道颜色
                        sidewalk_intersection = np.any(extended_rect == (250, 230, 230))  # 人行道颜色

                        if lane_intersection or sidewalk_intersection:
                            # 如果与车道或人行道相交，停止延伸
                            location_label = "Lane" if lane_intersection else "Sidewalk"
                            break

                        # 否则继续延伸
                        extended_y_bottom += 1

                    # 绘制延伸后的红色框
                    cv2.rectangle(img, (x_pred, y_pred), (x_pred + w_pred, extended_y_bottom), (0, 0, 255), 2)

                    # 在框外侧边显示位置标签
                    if location_label:
                        text_x = x_pred + w_pred + 5  # 在框右侧显示标签
                        text_y = y_pred + h_pred // 2  # 标签显示在框的垂直中心
                        cv2.putText(img, location_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 记录原始和延伸后的边框信息
                    missed_pedestrians.append({
                        'bounding_box': (x, y, w, h),
                        'iou': max_iou,
                        'location_label': location_label,
                        'extended_bounding_box': (x_pred, y_pred, w_pred, extended_y_bottom - y_pred)  # 记录延伸后的框
                    })

                    # 显示 IoU 值和“Missed”标签
                    cv2.putText(img, f"Missed (IoU={max_iou:.2f})", (x_pred, y_pred - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # 绘制预测轮廓（黄色）和真实轮廓（绿色）
                    cv2.drawContours(img, [gt_contour], -1, (0, 255, 0), 2)  # 绿色真实标签轮廓
                    cv2.drawContours(img, [best_pred_contour], -1, (0, 255, 255), 2)  # 黄色预测轮廓

        return missed_pedestrians, detected_pedestrians



    def compute_metrics(self, results: list) -> Dict[str, float]:
        """计算并返回评估指标"""
        total_missed = sum(result["missed_pedestrian_count"] for result in results)
        return {"total_missed_pedestrians": total_missed}


