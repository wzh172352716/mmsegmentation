from typing import Optional, Sequence, Dict

from mmengine import fileio
from mmengine.evaluator import BaseMetric

import numpy as np
import cv2
from mmseg.registry import METRICS
import collections
import os.path as osp

from PIL import Image

from mmseg.utils import datafrombytes

import torch
import torch.nn.functional as F  
# sbatch tools/slurm_test_ifn.sh configs/segformer/segformer_mit-b0_8xb1-160k_acdc-512x512.py /beegfs/work/bartels/mmsegmentation/downloaded_ckpts/segformer/segformer_mit-b0_8xb1-160k_acdc-512x512.py.pth test_evaluator.type="PedestrianSizeMetric"

@METRICS.register_module()
class PedestrianSizeMetric(BaseMetric):
    def __init__(self,
                 pedestrian_idx: int = 11,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            label_pedestrian = np.where(label == 11, 255, 0).astype("uint8")
            pred_label_pedestrian = np.where(pred_label == 11, 255, 0).astype("uint8")

            # cv2.imwrite("debug_label.png", label_pedestrian)

            ret, thresh = cv2.threshold(label_pedestrian, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            res = self.get_predicted_contours(contours, pred_label_pedestrian)

            for key, value in res.items():
                if key >= 200 and value["total_num"] > value["recognized_num"]:
                    basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
                    print(f"For image {basename} an pedestrian of large size (>=200px) was not predicted!")

                    out = cv2.cvtColor(label_pedestrian, cv2.COLOR_GRAY2RGB)
                    cv2.drawContours(out, contours, -1, (0, 0, 255), 3)

                    for key, value in res.items():
                        cv2.drawContours(out, value["contours"], -1, (0, 255, 0), 3)

                    cv2.imwrite(f"debug_{basename}.png", out)
            self.results.append(res)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        final_result = {}
        for result in results:
            for key, value in result.items():
                if not key in final_result:
                    final_result[key] = {"total_num": 0, "recognized_num": 0}
                final_result[key]["total_num"] += value["total_num"]
                final_result[key]["recognized_num"] += value["recognized_num"]
                # final_result[key]["contours"].extend(value["contours"])
        final_result = collections.OrderedDict(sorted(final_result.items()))
        print(final_result)
        return final_result

    def get_contour_height(self, contours):
        res = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            res.append(h)
            # cv2.putText(image, str(w), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
        return res

    def get_predicted_contours(self, contours, pred_mask):
        heights = self.get_contour_height(contours)
        res = {}
        # pred_mask_binary = np.where(pred_mask == 11, 255, 0).astype("uint8")

        for contour, height in zip(contours, heights):
            contour_img = np.zeros_like(pred_mask)
            cv2.drawContours(contour_img, [contour], -1, color=(255), thickness=cv2.FILLED)

            if not height in res:
                res[height] = {"total_num": 0, "recognized_num": 0, "contours": []}
            res[height]["total_num"] += 1
            intersection = cv2.bitwise_and(contour_img, pred_mask)
            # cv2.imwrite("intersection.png", intersection)
            if np.max(intersection) > 0:
                res[height]["recognized_num"] += 1
                res[height]["contours"].append(contour)

        return res


@METRICS.register_module()
class PixelSizeRecallMetric(BaseMetric):
    def __init__(self,
                 class_indices_label: list = [11],
                 class_indices_pred: list = [11, 12],
                 bin_width=1,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.class_indices_label = class_indices_label
        self.class_indices_pred = class_indices_pred
        self.bin_width = bin_width

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()

            label_pedestrian = np.zeros_like(label).astype("uint8")
            pred_label_pedestrian = np.zeros_like(pred_label).astype("uint8")
            for pred_idx in self.class_indices_pred:
                pred_label_pedestrian = np.bitwise_or(pred_label_pedestrian,
                                                      np.where(pred_label == pred_idx, 255, 0).astype("uint8"))
            for label_idx in self.class_indices_label:
                label_pedestrian = np.bitwise_or(label_pedestrian, np.where(label == label_idx, 255, 0).astype("uint8"))


            ret, thresh = cv2.threshold(label_pedestrian, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res = self.get_predicted_contours(contours, pred_label_pedestrian, label_pedestrian, data_sample)

            self.results.append(res)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        final_result = {}
        all_points = []

        for result in results:
            for height, instance_recall_list in result.items():
                if not height in final_result:
                    final_result[height] = {"total_num": 0, "mean_recall": 0}
                final_result[height]["mean_recall"] = (final_result[height]["mean_recall"] * final_result[height][
                    "total_num"] + sum(instance_recall_list)) / (final_result[height]["total_num"] + len(
                    instance_recall_list))
                final_result[height]["total_num"] += len(instance_recall_list)

        final_result = collections.OrderedDict(sorted(final_result.items()))
        # print(final_result)
        for result in results:
            for height, instance_recall_list in result.items():
                for recall in instance_recall_list:
                    all_points.append({height: recall})
        print("______________________________")
        print(all_points)

        return final_result

    def get_contour_height(self, contours):
        res = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            res.append(h)
            # cv2.putText(image, str(w), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
        return res

    def get_predicted_contours(self, contours, pred_mask, label_mask, data_sample):
        heights = self.get_contour_height(contours)
        res = {}

        basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
        save_img = False
        img = np.array(Image.open(data_sample['img_path']))
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wrong_instances = []

        for contour, height in zip(contours, heights):
            bin_height = (height // self.bin_width) * self.bin_width + 1
            contour_img = np.zeros_like(pred_mask)
            cv2.drawContours(contour_img, [contour], -1, color=(255), thickness=cv2.FILLED)
            contour_img = np.bitwise_and(label_mask, contour_img)

            if not bin_height in res:
                res[bin_height] = []

            negative_pred = np.ones_like(pred_mask) * 255 - pred_mask

            tp = np.sum(np.logical_and(contour_img == pred_mask, contour_img == np.ones_like(contour_img) * 255))
            fn = np.sum(np.logical_and(contour_img == negative_pred, contour_img == np.ones_like(contour_img) * 255))

            instance_recall = tp / (tp + fn)

            res[bin_height].append(instance_recall)

            if instance_recall <= 0.15 and height >= 100:
                save_img = True
                wrong_instances.append(contour)
                print(f"For image {basename} an pedestrian of size {height} was not predicted!")

        if save_img:
            cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
            cv2.drawContours(out, wrong_instances, -1, (0, 0, 255), 1)
            cv2.imwrite(f"debug_{basename}.png", out)

        return res

@METRICS.register_module()
class diou(BaseMetric):
    def __init__(self,
                 accuracyD=True,
                 accuracyI=True,
                 accuracyC=True,
                 q=10,
                 binary=False,
                 num_classes=19,  # for cityscapes
                 ignore_index=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.accuracyD = accuracyD
        self.accuracyI = accuracyI
        self.accuracyC = accuracyC
        self.q = q
        self.binary = binary
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tp = torch.tensor([], device=collect_device)  
        self.tn = torch.tensor([], device=collect_device)  
        self.fp = torch.tensor([], device=collect_device)  
        self.fn = torch.tensor([], device=collect_device)  
        self.active_classes = torch.tensor([], dtype=torch.bool, device=collect_device)  
        self.image_file = []  

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_sem_seg']['data'].squeeze().cpu()  
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu()  
            image_file = data_sample.get('img_path', None) 

            batch_size = pred.size(0)
            pred = pred.view(batch_size, -1)
            label = label.view(batch_size, -1)
            if self.ignore_index is not None:
                keep_mask = (label != self.ignore_index).to(torch.bool)
            else:
                keep_mask = torch.ones_like(label, dtype=torch.bool)
            keep_mask = keep_mask.unsqueeze(1).expand(batch_size, self.num_classes, -1)

            pred = F.one_hot(pred, num_classes=self.num_classes).permute(0, 2, 1)  
            label = torch.clamp(label, 0, self.num_classes - 1) 
            label = F.one_hot(label, num_classes=self.num_classes).permute(0, 2, 1)

            for i in range(batch_size):
                keep_mask_i = keep_mask[i, :, :]
                pred_i = pred[i, :, :][keep_mask_i].reshape(self.num_classes, -1)
                label_i = label[i, :, :][keep_mask_i].reshape(self.num_classes, -1)

                if label_i.size(1) < 1:
                    continue

                tp = torch.logical_and(pred_i == 1, label_i == 1)
                tn = torch.logical_and(pred_i == 0, label_i == 0)
                fp = torch.logical_and(pred_i == 1, label_i == 0)
                fn = torch.logical_and(pred_i == 0, label_i == 1)

                tp = torch.sum(tp, dim=1).unsqueeze(0)
                tn = torch.sum(tn, dim=1).unsqueeze(0)
                fp = torch.sum(fp, dim=1).unsqueeze(0)
                fn = torch.sum(fn, dim=1).unsqueeze(0)

                if self.binary:
                    mask = torch.amax(pred_i + label_i, dim=1) > 0.5
                else:
                    mask = torch.amax(label_i, dim=1) > 0.5

                mask = mask.unsqueeze(0)
                active_classes = torch.zeros(self.num_classes, dtype=torch.bool).unsqueeze(0)
                active_classes[mask] = 1

                self.tp = torch.cat((self.tp, tp), dim=0)
                self.tn = torch.cat((self.tn, tn), dim=0)
                self.fp = torch.cat((self.fp, fp), dim=0)
                self.fn = torch.cat((self.fn, fn), dim=0)
                self.active_classes = torch.cat((self.active_classes, active_classes), dim=0)
                self.image_file.append(image_file)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        final_results = {}
        if self.accuracyD:
            final_results.update(self.valueD())
        if self.accuracyI:
            final_results.update(self.valueI())
        if self.accuracyC:
            final_results.update(self.valueC())
        return final_results

    def valueD(self):
        tp = torch.sum(self.tp, dim=0)
        fp = torch.sum(self.fp, dim=0)
        fn = torch.sum(self.fn, dim=0)

        if self.binary:
            tp = tp[1]
            fp = fp[1]
            fn = fn[1]

        Acc = 100 * torch.sum(tp) / (torch.sum(tp + fn) + 1e-6)
        mAccD = 100 * torch.mean(tp / (tp + fn + 1e-6))
        mIoUD = 100 * torch.mean(tp / (tp + fp + fn + 1e-6))
        mDiceD = 100 * torch.mean(2 * tp / (2 * tp + fp + fn + 1e-6)) # Dice

        return {"Acc": Acc,
                "mAccD": mAccD,
                "mIoUD": mIoUD,
                "mDiceD": mDiceD}

    def valueI(self):
        AccIC = self.tp / (self.tp + self.fn + 1e-6)
        IoUIC = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        DiceIC = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-6)

        AccIC[~self.active_classes] = 0
        IoUIC[~self.active_classes] = 0
        DiceIC[~self.active_classes] = 0

        mAccI = self.reduceI(AccIC)
        mIoUI = self.reduceI(IoUIC)
        mDiceI = self.reduceI(DiceIC)

        mAccIQ = mIoUIQ = mDiceIQ = 0
        for q in range(10, 110, 10):
            mAccIQ += self.reduceI(AccIC, q)
            mIoUIQ += self.reduceI(IoUIC, q)
            mDiceIQ += self.reduceI(DiceIC, q)
        mAccIQ /= 10
        mIoUIQ /= 10
        mDiceIQ /= 10

        mAccIq = self.reduceI(AccIC, self.q)
        mIoUIq = self.reduceI(IoUIC, self.q)
        mDiceIq = self.reduceI(DiceIC, self.q)

        return {"mAccI": mAccI,
                "mIoUI": mIoUI,
                "mDiceI": mDiceI,
                "mAccIQ": mAccIQ,
                "mIoUIQ": mIoUIQ,
                "mDiceIQ": mDiceIQ,
                f"mAccI{self.q}": mAccIq,
                f"mIoUI{self.q}": mIoUIq,
                f"mDiceI{self.q}": mDiceIq}

    def valueC(self):
        AccIC = self.tp / (self.tp + self.fn + 1e-6)
        IoUIC = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        DiceIC = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-6)

        AccIC[~self.active_classes] = 1e6
        IoUIC[~self.active_classes] = 1e6
        DiceIC[~self.active_classes] = 1e6

        mAccC = self.reduceC(AccIC)
        mIoUC = self.reduceC(IoUIC)
        mDiceC = self.reduceC(DiceIC)

        mAccCQ = mIoUCQ = mDiceCQ = 0
        for q in range(10, 110, 10):
            mAccCQ += self.reduceC(AccIC, q)
            mIoUCQ += self.reduceC(IoUIC, q)
            mDiceCQ += self.reduceC(DiceIC, q)
        mAccCQ /= 10
        mIoUCQ /= 10
        mDiceCQ /= 10

        mAccCq = self.reduceC(AccIC, self.q)
        mIoUCq = self.reduceC(IoUIC, self.q)
        mDiceCq = self.reduceC(DiceIC, self.q)

        return {"mAccC": mAccC,
                "mIoUC": mIoUC,
                "mDiceC": mDiceC,
                "mAccCQ": mAccCQ,
                "mIoUCQ": mIoUCQ,
                "mDiceCQ": mDiceCQ,
                f"mAccC{self.q}": mAccCq,
                f"mIoUC{self.q}": mIoUCq,
                f"mDiceC{self.q}": mDiceCq}

    def reduceI(self, value_matrix, q=None):
        active_sum = torch.sum(self.active_classes, dim=1)
        if self.binary:
            value = value_matrix[:, 1]
            value[active_sum < 2] = 1
        else:
            value = torch.sum(value_matrix, dim=1)
            value /= active_sum

        if q is None:
            n = value.size(0)
        else:
            n = max(1, int(q / 100 * value.size(0)))

        # # 获取排序后的索引
        # sorted_indices = torch.argsort(value)[:n]
        # sorted_value = value[sorted_indices]

        # # 输出排序后的索引和对应的图片路径
        # print(f"Indices of worst {q}% samples: {sorted_indices}")
        # for idx in sorted_indices:
        #     print(f"Image file: {self.image_file[idx]}, Value: {sorted_value[idx]}")
        
        value = torch.sort(value)[0][:n]
        value = 100 * torch.mean(value)

        return value

    def reduceC(self, value_matrix, q=None):
        num_images, num_classes = value_matrix.shape
        active_sum = torch.sum(self.active_classes, dim=0)
        if q is not None:
            active_sum = torch.max(torch.ones(num_classes), (q / 100 * active_sum).to(torch.long))

        indices = torch.arange(num_images).view(-1, 1).expand_as(value_matrix)
        mask = indices < active_sum

        value_matrix = mask * torch.sort(value_matrix, dim=0)[0]
        value = torch.sum(value_matrix, dim=0) / active_sum
        value = 100 * torch.mean(value)

        return value

@METRICS.register_module()
class PedestrianDistanceMetric(BaseMetric):
    def __init__(self,
                 pedestrian_idx: int = 11,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 fx: float = 2262.52,  # 水平方向焦距
                 fy: float = 2265.30,  # 垂直方向焦距
                 image_height: int = 1024,  # 图像的垂直分辨率
                 known_height: float = 1.7,  # 行人的实际高度
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pedestrian_idx = pedestrian_idx
        self.fx = fx
        self.fy = fy
        self.image_height = image_height
        self.known_height = known_height
        self.results = []
        self.image_info = []  # 用于存储每张图像的相关信息

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        处理每批数据样本，提取预测和真实标签，计算每个行人的轮廓高度和距离
        """
        for data_sample in data_samples:
            # 获取预测和真实标签
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            image_path = data_sample['img_path']  # 获取图像路径
            image_name = osp.basename(image_path)
            image = cv2.imread(image_path)  # 读取原始图像

            # 提取行人区域
            label_pedestrian = np.where(label == self.pedestrian_idx, 255, 0).astype("uint8")
            pred_label_pedestrian = np.where(pred_label == self.pedestrian_idx, 255, 0).astype("uint8")

            # 获取真实标签中的行人轮廓
            ret, thresh = cv2.threshold(label_pedestrian, 127, 255, cv2.THRESH_BINARY)
            gt_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 获取预测结果中的行人轮廓
            ret_pred, thresh_pred = cv2.threshold(pred_label_pedestrian, 127, 255, cv2.THRESH_BINARY)
            pred_contours, hierarchy_pred = cv2.findContours(thresh_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 存储当前图像的行人高度信息（使用预测轮廓）
            contour_heights = []
            contour_distances = []
            for contour in pred_contours:
                x, y, w, h = cv2.boundingRect(contour)
                contour_heights.append(h)
                distance = self.calculate_distance_monocular(h)
                contour_distances.append(distance)

            # 如果存在预测的行人，记录最大和最小高度及对应的距离和轮廓
            if contour_heights:
                max_height = max(contour_heights)
                min_height = min(contour_heights)
                max_index = contour_heights.index(max_height)
                min_index = contour_heights.index(min_height)
                max_distance = contour_distances[max_index]
                min_distance = contour_distances[min_index]
                max_contour = pred_contours[max_index]
                min_contour = pred_contours[min_index]

                # 存储信息
                self.image_info.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'image': image,
                    'gt_contours': gt_contours,
                    'pred_contours': pred_contours,
                    'max_height': max_height,
                    'min_height': min_height,
                    'max_distance': max_distance,
                    'min_distance': min_distance,
                    'max_contour': max_contour,
                    'min_contour': min_contour,
                })
            else:
                # 如果没有预测到行人，仍然保存图像信息，以便分析
                self.image_info.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'image': image,
                    'gt_contours': gt_contours,
                    'pred_contours': [],
                    'max_height': 0,
                    'min_height': 0,
                    'max_distance': 0,
                    'min_distance': 0,
                    'max_contour': None,
                    'min_contour': None,
                })

    def calculate_distance_monocular(self, pixel_height):
        """
        基于单目摄像机的行人像素高度计算距离
        :param pixel_height: 行人的像素高度
        :return: 估算的距离 (米)
        """
        if pixel_height == 0:
            return float('inf')  # 避免除以零
        distance = (self.known_height * self.fy) / pixel_height
        return distance

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        计算累计的行人检测指标，并输出指定的结果图像
        """
        # 按照最大高度排序，选取五张具有最大行人像素高度的图像
        sorted_by_max_height = sorted(self.image_info, key=lambda x: x['max_height'], reverse=True)
        top_5_max = sorted_by_max_height[:5]

        # 按照最小高度排序，选取五张具有最小行人像素高度的图像（大于0）
        sorted_by_min_height = sorted([info for info in self.image_info if info['min_height'] > 0], key=lambda x: x['min_height'])
        top_5_min = sorted_by_min_height[:5]

        # 合并这十张图像信息
        selected_images = top_5_max + top_5_min

        # 对于每张图像，绘制真实和预测的行人轮廓以及距离
        for idx, info in enumerate(selected_images):
            image = info['image'].copy()
            image_name = info['image_name']

            # 绘制真实的行人轮廓（红色）
            if info['gt_contours']:
                cv2.drawContours(image, info['gt_contours'], -1, (0, 0, 255), 2)

            # 绘制预测的行人轮廓（绿色）
            if info['pred_contours']:
                cv2.drawContours(image, info['pred_contours'], -1, (0, 255, 0), 2)

            # 绘制最大和最小预测行人轮廓并标注距离
            if info['max_contour'] is not None:
                self.annotate_image(image, [info['max_contour']], [info['max_distance']], (0, 255, 0), 'Max Distance')
            if info['min_contour'] is not None and info['min_contour'] is not info['max_contour']:
                self.annotate_image(image, [info['min_contour']], [info['min_distance']], (255, 0, 0), 'Min Distance')

            # 保存结果图像
            result_image_name = f'result_{idx+1}_{image_name}'
            cv2.imwrite(result_image_name, image)
            print(f'Saved annotated image: {result_image_name}')

        return {}  # 返回需要的指标结果

    def annotate_image(self, image, contours, distances, color, label):
        """
        在图像上绘制轮廓和标注距离
        """
        for contour, distance in zip(contours, distances):
            if contour is not None:
                cv2.drawContours(image, [contour], -1, color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(image, f'{label}: {distance:.2f}m', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def get_contour_height(self, contours):
        """
        获取轮廓的高度
        """
        heights = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            heights.append(h)
        return heights

    def get_predicted_contours(self, contours, pred_mask):
        """
        获取预测的轮廓，并统计每个轮廓的检测情况
        """
        heights = self.get_contour_height(contours)
        res = {}

        for contour, height in zip(contours, heights):
            contour_img = np.zeros_like(pred_mask)
            cv2.drawContours(contour_img, [contour], -1, color=(255), thickness=cv2.FILLED)

            if height not in res:
                res[height] = {"total_num": 0, "recognized_num": 0, "contours": []}
            res[height]["total_num"] += 1
            intersection = cv2.bitwise_and(contour_img, pred_mask)
            if np.max(intersection) > 0:
                res[height]["recognized_num"] += 1
                res[height]["contours"].append(contour)

        return res

@METRICS.register_module()
class WorstPedestrain(BaseMetric):
    def __init__(self,
                 accuracyD=True,
                 accuracyI=True,
                 accuracyC=True,
                 q=10,
                 binary=False,
                 num_classes=19,  # for Cityscapes
                 pedestrian_class_index=11,  # 行人类别的索引
                 ignore_index=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.accuracyD = accuracyD
        self.accuracyI = accuracyI
        self.accuracyC = accuracyC
        self.q = q
        self.binary = binary
        self.num_classes = num_classes
        self.pedestrian_class_index = pedestrian_class_index  # 行人类别索引
        self.ignore_index = ignore_index
        self.tp = torch.tensor([], device=collect_device)
        self.tn = torch.tensor([], device=collect_device)
        self.fp = torch.tensor([], device=collect_device)
        self.fn = torch.tensor([], device=collect_device)
        self.active_classes = torch.tensor([], dtype=torch.bool, device=collect_device)
        self.image_file = []
        self.contains_pedestrian = []  # 记录每个样本是否包含行人
        # 新增以下列表用于存储每个样本的预测和真实标签
        self.predictions = []
        self.labels = []    

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_sem_seg']['data'].squeeze().cpu()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu()
            image_file = data_sample.get('img_path', None)

            # 保存原始预测和标签，用于后续保存图像
            self.predictions.append(pred.numpy())
            self.labels.append(label.numpy())
            self.image_file.append(image_file)

            batch_size = pred.size(0)
            pred = pred.view(batch_size, -1)
            label = label.view(batch_size, -1)
            if self.ignore_index is not None:
                keep_mask = (label != self.ignore_index).to(torch.bool)
            else:
                keep_mask = torch.ones_like(label, dtype=torch.bool)
            keep_mask = keep_mask.view(batch_size, -1)

            # 将标签值限制在有效范围内
            label = torch.clamp(label, 0, self.num_classes - 1)

            # 将预测和标签转换为独热编码
            pred = F.one_hot(pred, num_classes=self.num_classes)
            label = F.one_hot(label, num_classes=self.num_classes)

            # 转置张量以匹配形状 [batch_size, num_classes, N]
            pred = pred.permute(0, 2, 1)
            label = label.permute(0, 2, 1)

            for i in range(batch_size):
                # 对当前样本应用keep_mask
                keep_mask_i = keep_mask[i, :]
                pred_i = pred[i, :, keep_mask_i]  # [num_classes, valid_pixels]
                label_i = label[i, :, keep_mask_i]  # [num_classes, valid_pixels]

                if label_i.size(1) < 1:
                    continue

                # 计算TP、FP、FN、TN
                tp = (pred_i == 1) & (label_i == 1)
                tn = (pred_i == 0) & (label_i == 0)
                fp = (pred_i == 1) & (label_i == 0)
                fn = (pred_i == 0) & (label_i == 1)

                tp = tp.sum(dim=1).unsqueeze(0)  # [1, num_classes]
                tn = tn.sum(dim=1).unsqueeze(0)
                fp = fp.sum(dim=1).unsqueeze(0)
                fn = fn.sum(dim=1).unsqueeze(0)

                # 确定在当前样本中激活的类别
                if self.binary:
                    mask = (pred_i + label_i).amax(dim=1) > 0.5
                else:
                    mask = label_i.amax(dim=1) > 0.5
                mask = mask.unsqueeze(0)
                active_classes = torch.zeros(1, self.num_classes, dtype=torch.bool)
                active_classes[0] = mask

                # 检查样本是否包含行人类别
                contains_pedestrian = (label_i[self.pedestrian_class_index] > 0).any().item()
                self.contains_pedestrian.append(contains_pedestrian)

                # 收集指标
                self.tp = torch.cat((self.tp, tp), dim=0)
                self.tn = torch.cat((self.tn, tn), dim=0)
                self.fp = torch.cat((self.fp, fp), dim=0)
                self.fn = torch.cat((self.fn, fn), dim=0)
                self.active_classes = torch.cat((self.active_classes, active_classes), dim=0)
                self.image_file.append(image_file)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        final_results = {}
        if self.accuracyD:
            final_results.update(self.valueD())
        if self.accuracyI:
            final_results.update(self.valueI())
        if self.accuracyC:
            final_results.update(self.valueC())
        # 计算并保存最差的 q% 样本
        self.save_worst_samples()
        return final_results

    def valueD(self):
        tp = torch.sum(self.tp, dim=0)
        fp = torch.sum(self.fp, dim=0)
        fn = torch.sum(self.fn, dim=0)

        # 计算行人类别的指标
        tp_pedestrian = tp[self.pedestrian_class_index]
        fp_pedestrian = fp[self.pedestrian_class_index]
        fn_pedestrian = fn[self.pedestrian_class_index]

        IoU_pedestrian = tp_pedestrian / (tp_pedestrian + fp_pedestrian + fn_pedestrian + 1e-6)
        mIoU_pedestrian = 100 * IoU_pedestrian

        # 计算整体指标（可选）
        if self.binary:
            tp = tp[1]
            fp = fp[1]
            fn = fn[1]

        Acc = 100 * torch.sum(tp) / (torch.sum(tp + fn) + 1e-6)
        mAccD = 100 * torch.mean(tp / (tp + fn + 1e-6))
        mIoUD = 100 * torch.mean(tp / (tp + fp + fn + 1e-6))
        mDiceD = 100 * torch.mean(2 * tp / (2 * tp + fp + fn + 1e-6))  # Dice

        return {"Acc": Acc,
                "mAccD": mAccD,
                "mIoUD": mIoUD,
                "mDiceD": mDiceD,
                "mIoU_pedestrian": mIoU_pedestrian}

    def valueI(self):
        AccIC = self.tp / (self.tp + self.fn + 1e-6)
        IoUIC = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        DiceIC = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-6)

        AccIC[~self.active_classes] = 0
        IoUIC[~self.active_classes] = 0
        DiceIC[~self.active_classes] = 0

        # 提取行人类别的指标
        pedestrian_class_index = self.pedestrian_class_index
        Acc_pedestrian = AccIC[:, pedestrian_class_index]
        IoU_pedestrian = IoUIC[:, pedestrian_class_index]
        Dice_pedestrian = DiceIC[:, pedestrian_class_index]

        # 仅考虑包含行人类别的样本
        active_pedestrian = self.active_classes[:, pedestrian_class_index]
        Acc_pedestrian = Acc_pedestrian[active_pedestrian]
        IoU_pedestrian = IoU_pedestrian[active_pedestrian]
        Dice_pedestrian = Dice_pedestrian[active_pedestrian]

        # 计算行人类别的平均指标
        if len(IoU_pedestrian) > 0:
            mAccI_pedestrian = 100 * torch.mean(Acc_pedestrian)
            mIoUI_pedestrian = 100 * torch.mean(IoU_pedestrian)
            mDiceI_pedestrian = 100 * torch.mean(Dice_pedestrian)
        else:
            mAccI_pedestrian = mIoUI_pedestrian = mDiceI_pedestrian = 0.0

        # 计算整体指标
        mAccI = self.reduceI(AccIC)
        mIoUI = self.reduceI(IoUIC)
        mDiceI = self.reduceI(DiceIC)

        mAccIQ = mIoUIQ = mDiceIQ = 0
        for q in range(10, 110, 10):
            mAccIQ += self.reduceI(AccIC, q)
            mIoUIQ += self.reduceI(IoUIC, q)
            mDiceIQ += self.reduceI(DiceIC, q)
        mAccIQ /= 10
        mIoUIQ /= 10
        mDiceIQ /= 10

        mAccIq = self.reduceI(AccIC, self.q)
        mIoUIq = self.reduceI(IoUIC, self.q)
        mDiceIq = self.reduceI(DiceIC, self.q)
        mIoUI_pedestrian_q = self.reduceI_class(IoUIC, self.pedestrian_class_index, q=self.q)
        return {"mAccI": mAccI,
                "mIoUI": mIoUI,
                "mDiceI": mDiceI,
                "mAccIQ": mAccIQ,
                "mIoUIQ": mIoUIQ,
                "mDiceIQ": mDiceIQ,
                f"mAccI{self.q}": mAccIq,
                f"mIoUI{self.q}": mIoUIq,
                f"mDiceI{self.q}": mDiceIq,
                "mAccI_pedestrian": mAccI_pedestrian,
                "mIoUI_pedestrian": mIoUI_pedestrian,
                "mDiceI_pedestrian": mDiceI_pedestrian,
                f"mIoUI_pedestrian_{self.q}": mIoUI_pedestrian_q
                }

    def valueC(self):
        AccIC = self.tp / (self.tp + self.fn + 1e-6)
        IoUIC = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        DiceIC = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-6)

        AccIC[~self.active_classes] = 1e6
        IoUIC[~self.active_classes] = 1e6
        DiceIC[~self.active_classes] = 1e6

        # 提取行人类别的指标
        pedestrian_class_index = self.pedestrian_class_index
        Acc_pedestrian = AccIC[:, pedestrian_class_index]
        IoU_pedestrian = IoUIC[:, pedestrian_class_index]
        Dice_pedestrian = DiceIC[:, pedestrian_class_index]

        # 仅考虑包含行人类别的样本
        active_pedestrian = self.active_classes[:, pedestrian_class_index]
        Acc_pedestrian = Acc_pedestrian[active_pedestrian]
        IoU_pedestrian = IoU_pedestrian[active_pedestrian]
        Dice_pedestrian = Dice_pedestrian[active_pedestrian]

        # 计算行人类别的平均指标
        if len(IoU_pedestrian) > 0:
            mAccC_pedestrian = 100 * torch.mean(Acc_pedestrian)
            mIoUC_pedestrian = 100 * torch.mean(IoU_pedestrian)
            mDiceC_pedestrian = 100 * torch.mean(Dice_pedestrian)
        else:
            mAccC_pedestrian = mIoUC_pedestrian = mDiceC_pedestrian = 0.0

        # 计算整体指标
        mAccC = self.reduceC(AccIC)
        mIoUC = self.reduceC(IoUIC)
        mDiceC = self.reduceC(DiceIC)

        mAccCQ = mIoUCQ = mDiceCQ = 0
        for q in range(10, 110, 10):
            mAccCQ += self.reduceC(AccIC, q)
            mIoUCQ += self.reduceC(IoUIC, q)
            mDiceCQ += self.reduceC(DiceIC, q)
        mAccCQ /= 10
        mIoUCQ /= 10
        mDiceCQ /= 10

        mAccCq = self.reduceC(AccIC, self.q)
        mIoUCq = self.reduceC(IoUIC, self.q)
        mDiceCq = self.reduceC(DiceIC, self.q)

        return {"mAccC": mAccC,
                "mIoUC": mIoUC,
                "mDiceC": mDiceC,
                "mAccCQ": mAccCQ,
                "mIoUCQ": mIoUCQ,
                "mDiceCQ": mDiceCQ,
                f"mAccC{self.q}": mAccCq,
                f"mIoUC{self.q}": mIoUCq,
                f"mDiceC{self.q}": mDiceCq,
                "mAccC_pedestrian": mAccC_pedestrian,
                "mIoUC_pedestrian": mIoUC_pedestrian,
                "mDiceC_pedestrian": mDiceC_pedestrian}

    def reduceI(self, value_matrix, q=None):
        active_sum = torch.sum(self.active_classes, dim=1)
        if self.binary:
            value = value_matrix[:, 1]
            value[active_sum < 2] = 1
        else:
            value = torch.sum(value_matrix, dim=1) / (active_sum + 1e-6)

        pedestrian_class_index = self.pedestrian_class_index

        # 转换为布尔张量，标记每个图像是否包含行人
        contains_pedestrian = torch.tensor(self.contains_pedestrian, dtype=torch.bool, device=value.device)
        # 判断模型是否在该图像中预测了行人类别
        predicted_pedestrian = (self.tp[:, pedestrian_class_index] + self.fp[:, pedestrian_class_index]) > 0

        # 筛选有效的样本（实际包含行人或模型预测了行人）
        valid_indices = contains_pedestrian | predicted_pedestrian

        # 过滤有效样本的值
        value = value[valid_indices]
        if value.numel() == 0:
            return 0.0  # 无有效样本可计算

        # 计算需要选取的最差样本数量n
        if q is None:
            n = value.size(0)
        else:
            n = max(1, int(q / 100 * value.size(0)))

        # 对值进行排序，选取最差的n个样本
        sorted_value, sorted_indices = torch.sort(value)
        value = sorted_value[:n]

        # 计算最差n个样本的平均值
        value = 100 * torch.mean(value)

        return value

    def reduceC(self, value_matrix, q=None):
        num_images, num_classes = value_matrix.shape
        active_sum = torch.sum(self.active_classes, dim=0)
        if q is not None:
            active_sum = torch.max(torch.ones(num_classes), (q / 100 * active_sum).to(torch.long))

        indices = torch.arange(num_images).view(-1, 1).expand_as(value_matrix)
        mask = indices < active_sum

        value_matrix = mask * torch.sort(value_matrix, dim=0)[0]
        value = torch.sum(value_matrix, dim=0) / active_sum
        value = 100 * torch.mean(value)

        return value

    def reduceI_class(self, value_matrix, class_index, q=None):
        # 提取指定类别的指标
        value = value_matrix[:, class_index]

        # 转换为布尔张量，标记每个图像是否包含该类别
        active_class = self.active_classes[:, class_index]

        # 筛选有效的样本（实际包含该类别的样本）
        value = value[active_class]
        if value.numel() == 0:
            return 0.0  # 无有效样本可计算

        # 计算需要选取的最差样本数量 n
        if q is None:
            n = value.size(0)
        else:
            n = max(1, int(q / 100 * value.size(0)))

        # 对值进行排序，选取最差的 n 个样本
        sorted_value, sorted_indices = torch.sort(value)
        value = sorted_value[:n]

        # 计算最差 n 个样本的平均值
        value = 100 * torch.mean(value)
        return value

    def save_worst_samples(self):
        # 确保保存结果的文件夹存在
        save_dir = 'worstresults'
        os.makedirs(save_dir, exist_ok=True)

        # 计算每个样本的行人类别 IoU
        IoUIC = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        pedestrian_class_index = self.pedestrian_class_index
        IoU_pedestrian = IoUIC[:, pedestrian_class_index]

        # 转换为 CPU numpy 数组
        IoU_pedestrian = IoU_pedestrian.cpu().numpy()
        contains_pedestrian = np.array(self.contains_pedestrian)

        # 仅考虑包含行人类别的样本
        valid_indices = np.where(contains_pedestrian)[0]
        if valid_indices.size == 0:
            print("No samples contain the pedestrian class.")
            return

        IoU_pedestrian = IoU_pedestrian[valid_indices]
        image_files = [self.image_file[i] for i in valid_indices]
        predictions = [self.predictions[i] for i in valid_indices]
        labels = [self.labels[i] for i in valid_indices]

        # 计算需要选取的最差样本数量 n
        q = self.q
        n = max(1, int(q / 100 * len(IoU_pedestrian)))

        # 对 IoU 值进行排序，选取最差的 n 个样本
        sorted_indices = np.argsort(IoU_pedestrian)
        worst_indices = sorted_indices[:n]

        # 保存最差的样本
        for idx in worst_indices:
            image_file = image_files[idx]
            pred = predictions[idx]
            label = labels[idx]
            IoU_value = IoU_pedestrian[idx]

            # 读取原始图像
            if image_file is not None and os.path.exists(image_file):
                image = cv2.imread(image_file)
            else:
                print(f"Image file {image_file} not found.")
                continue

            # 创建彩色的预测结果和真实标签
            pred_color = self.colorize_mask(pred)
            label_color = self.colorize_mask(label)

            # 叠加预测结果和真实标签到原始图像
            overlay_pred = cv2.addWeighted(image, 0.6, pred_color, 0.4, 0)
            overlay_label = cv2.addWeighted(image, 0.6, label_color, 0.4, 0)

            # 合并显示
            combined_image = np.hstack((overlay_pred, overlay_label))

            # 在图像上标注 IoU 值
            cv2.putText(combined_image, f"IoU: {IoU_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 保存图像
            basename = os.path.basename(image_file)
            save_path = os.path.join(save_dir, f"worst_{basename}")
            cv2.imwrite(save_path, combined_image)
            print(f"Saved worst sample image: {save_path}")
    
    def colorize_mask(self, mask):
        palette = self.get_palette(self.num_classes)
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label_index in range(self.num_classes):
            color = palette[label_index]
            color_mask[mask == label_index] = color
        return color_mask
    def get_palette(self, num_classes):
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for i in range(num_classes):
            palette[i] = [i * (255 // num_classes), 255 - i * (255 // num_classes), (i * 37) % 255]
        return palette