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

import shutil
from mmengine.logging import MMLogger
import os
import json
from numpy.typing import NDArray

def mkdir_or_exist(dir_name):
    """如果目录不存在，则创建该目录"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
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
    
# 根据行人的边界画框。计算框中人的像素面积和车道像素面积之比，结合老版距离计算。
#问题：1 未判断iou大小。应该加上判断iou>0.3 这样才是对于已检出行人的位置判断
#     2 距离计算用的未更新的版本，改成最新版本的距离计算
@METRICS.register_module()
class LaneSegmentationMetric(BaseMetric):
    def __init__(self,
                 lane_idx: int = 0,  # 车道类标签索引
                 pedestrian_idx: int = 11,  # 行人类标签索引
                 min_area_threshold: int = 100,  # 最小车道面积阈值
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 output_dir: Optional[str] = './high_lane_intersection_images',
                 fx: float = 2262.52,  # 水平方向焦距
                 fy: float = 2265.30,  # 垂直方向焦距
                 known_height: float = 1.7,  # 行人的实际高度
                 min_width: int = 10,  # 行人最小宽度（像素）
                 min_height: int = 10,  # 行人最小高度（像素）
                 **kwargs) -> None:
        
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        # 行人和车道的索引、阈值
        self.lane_idx = lane_idx
        self.pedestrian_idx = pedestrian_idx
        self.min_area_threshold = min_area_threshold
        
        # 图像输出参数
        self.output_dir = osp.abspath(output_dir) if output_dir else None
        self.output_orig_dir = osp.join(self.output_dir, 'originals') if self.output_dir else None
        self.output_visual_dir = osp.join(self.output_dir, 'visualized') if self.output_dir else None
        if self.output_dir:
            mkdir_or_exist(self.output_dir)
            mkdir_or_exist(self.output_orig_dir)
            mkdir_or_exist(self.output_visual_dir)

        # 保存文件路径和数量的控制
        self.saved_image_paths = []
        self.max_images_to_save = 100  # 将图片保存数量限制改为 100
        
        # 距离计算参数
        self.fx = fx
        self.fy = fy
        self.known_height = known_height

        # 最小宽度和高度，用于过滤小行人边界框
        self.min_width = min_width
        self.min_height = min_height
        
        # 存储结果
        self.results = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        logger = MMLogger.get_current_instance()

        for data_sample in data_samples:
            # 获取图像的基础名称和路径
            img_path = data_sample['img_path']
            basename = osp.splitext(osp.basename(img_path))[0]

            # 提取预测和真实标签的车道和行人区域
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()

            label_lane = self.extract_mask(label, self.lane_idx)
            pred_label_lane = self.extract_mask(pred_label, self.lane_idx)
            label_pedestrian = self.extract_mask(label, self.pedestrian_idx)
            pred_label_pedestrian = self.extract_mask(pred_label, self.pedestrian_idx)

            # 读取原始图像
            original_img = cv2.imread(img_path)

            # 调用 visualize_intersection 计算行人重叠信息并直接标注到图像
            self.visualize_intersection(
                original_img, label_lane, pred_label_lane, label_pedestrian, pred_label_pedestrian
            )

            # 随机选择并保存图像，传递带有标注的图像对象
            if self.output_dir is not None and len(self.saved_image_paths) < self.max_images_to_save:
                self.save_image(original_img, basename, img_path)  # 传递标注后的 original_img

            # 计算并记录 IoU
            self.results.append({"lane_iou": self.compute_iou(label_lane, pred_label_lane)})


    def save_image(self, visual_img, basename, img_path):   
        """保存原始和带标注的图像"""
        mkdir_or_exist(self.output_orig_dir)
        mkdir_or_exist(self.output_visual_dir)
        
        # 保存原始图像
        orig_save_path = osp.join(self.output_orig_dir, f"{basename}_original.png")
        shutil.copy(img_path, orig_save_path)

        # 保存标注后的图像
        visual_save_path = osp.join(self.output_visual_dir, f"{basename}_occlusion_visualized.png")
        cv2.imwrite(visual_save_path, visual_img)

        self.saved_image_paths.append((orig_save_path, visual_save_path))


    def extract_mask(self, label, idx):
        """提取特定索引的二值掩码"""
        return np.where(label == idx, 255, 0).astype("uint8")
    
    def find_contours(self, mask):
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def visualize_intersection(self, img, label_lane, pred_label_lane, label_pedestrian, pred_label_pedestrian):
        """在图像上标出车道区域和行人区域，并标注距离和重叠率信息"""

        min_width, min_height = 10, 10  # 设置最小宽度和高度阈值

        img[label_lane == 255] = (255, 0, 0)  # 蓝色填充真实车道区域
        img[pred_label_lane == 255] = (0, 255, 0)  # 绿色填充预测车道区域
        img[pred_label_pedestrian == 255] = (0, 0, 255)  # 红色填充预测行人区域

        # 计算每个行人区域的重叠率
        contours_pedestrian = self.find_contours(pred_label_pedestrian)
        for contour in contours_pedestrian:
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤掉较小的行人区域
            if w < min_width or h < min_height:
                continue
            # 用标注颜色过后的面积进行计算？
            # roi = img[y:y+h, x:x+w]
            # red_area = np.sum(np.all(roi == [0, 0, 255], axis=-1))
            # green_area = np.sum(np.all(roi == [0, 255, 0], axis=-1))
            # overlap_ratio = green_area / red_area if red_area > 0 else 0
            pedestrian_roi = pred_label_pedestrian[y:y+h, x:x+w]
            lane_roi = pred_label_lane[y:y+h, x:x+w]
            overlap_ratio = self.calculate_overlap_ratio(pedestrian_roi, lane_roi)
            # 计算距离
            distance = self.calculate_distance_monocular(h)  # 使用高度 h 计算距离

            # 设置框颜色
            box_color = (0, 255, 0) if overlap_ratio < 0.05 else (0, 255, 255)  # 绿色或黄色框

            # 在图像上标注距离和重叠率
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(img, f"Ratio: {overlap_ratio:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
            cv2.putText(img, f"Distance: {distance:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)


    def calculate_distance_monocular(self, pixel_height):
        """基于单目摄像机的行人像素高度计算距离"""
        if pixel_height == 0:
            return float('inf')  # 避免除以零
        distance = (self.known_height * self.fy) / pixel_height
        return distance
    

    def compute_iou(self, label_mask, pred_mask):
        """计算 IoU 指标"""
        intersection = np.sum((label_mask == 255) & (pred_mask == 255))
        union = np.sum((label_mask == 255) | (pred_mask == 255))
        return intersection / union if union > 0 else 0.0
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """计算并返回评估指标，必要时实现具体逻辑"""
        # 这里可以添加具体的指标计算逻辑
        return {}
    def calculate_overlap_ratio(self, pedestrian_mask: NDArray, lane_mask: NDArray) -> float:
        intersection = np.logical_and(pedestrian_mask == 255, lane_mask == 255).sum()
        pedestrian_area = (pedestrian_mask == 255).sum()
        return intersection / pedestrian_area if pedestrian_area > 0 else 0

# 漏检行人的位置 检测iou在0.01-0.3之间的人定义为漏检人群
@METRICS.register_module()
class MissedPedestrianMetric(BaseMetric):
    def __init__(self,
                 lane_idx: int = 0,
                 sidewalk_idx: int = 1,
                 pedestrian_idx: int = 11,
                 min_area_threshold: int = 100,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 output_dir: Optional[str] = './missed_pedestrian_images',
                 min_width: int = 10,
                 min_height: int = 10,
                 max_images_to_save: int = 100,
                 iou_threshold_min: float = 0.02,
                 iou_threshold_max: float = 0.3,
                 **kwargs) -> None:
        
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.lane_idx = lane_idx
        self.sidewalk_idx = sidewalk_idx
        self.pedestrian_idx = pedestrian_idx
        self.min_area_threshold = min_area_threshold
        self.iou_threshold_min = iou_threshold_min
        self.iou_threshold_max = iou_threshold_max
        
        # 图像输出参数
        self.output_dir = osp.abspath(output_dir) if output_dir else None
        self.output_orig_dir = osp.join(self.output_dir, 'originals') if self.output_dir else None
        self.output_visual_dir = osp.join(self.output_dir, 'visualized') if self.output_dir else None
        if self.output_dir:
            mkdir_or_exist(self.output_dir)
            mkdir_or_exist(self.output_orig_dir)
            mkdir_or_exist(self.output_visual_dir)

        # 控制保存文件数量
        self.saved_image_paths = []
        self.max_images_to_save = max_images_to_save

        # 最小宽度和高度
        self.min_width = min_width
        self.min_height = min_height
        
        # 存储结果
        self.missed_pedestrian_dict = {}
        self.results = []


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

            # 查找漏检的行人并在图像上标记
            missed_pedestrians = self.visualize_missed_pedestrians(original_img, label_pedestrian, pred_label_pedestrian, pred_label_lane, pred_label_sidewalk)


            # 将漏检行人信息添加到字典中
            if missed_pedestrians:
                self.missed_pedestrian_dict[basename] = missed_pedestrians

                # 仅在存在漏检行人时保存图像
                if self.output_dir is not None and len(self.saved_image_paths) < self.max_images_to_save:
                    self.save_image(original_img, basename, img_path)


            # 将漏检行人数添加到 self.results
            self.results.append({"missed_pedestrian_count": len(missed_pedestrians)})

        # 将字典保存到 JSON 文件
        output_file_path = osp.join(self.output_dir, "missed_pedestrians.json")
        with open(output_file_path, 'w') as f:
            json.dump(self.missed_pedestrian_dict, f, indent=4)



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


    def visualize_missed_pedestrians(self, img, label_pedestrian, pred_label_pedestrian, pred_label_lane, pred_label_sidewalk):
        """在图像上标出漏检的行人区域，仅在满足 IoU 在 0 到 0.3 之间时绘制红框和轮廓"""

        missed_pedestrians = []

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

            # 初始化最大 IoU 为 0
            max_iou = 0
            best_pred_contour = None
            for pred_contour in pred_contours:
                # 计算 GT 和 Pred 轮廓之间的 IoU
                iou = self.compute_iou_contours(gt_contour, pred_contour, label_pedestrian.shape)
                if iou > max_iou:
                    max_iou = iou
                    best_pred_contour = pred_contour  # 保存 IoU 最大的预测轮廓

            # 只保留 IoU 在 0.02 到 0.3 之间的行人
            if self.iou_threshold_min < max_iou <= self.iou_threshold_min:
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

                    # 打印调试信息
                    if lane_intersection:
                        print(f"Pedestrian in lane at ({x_pred}, {extended_y_bottom})")
                    elif sidewalk_intersection:
                        print(f"Pedestrian on sidewalk at ({x_pred}, {extended_y_bottom})")
                    else:
                        print(f"Pedestrian not on lane or sidewalk")

        return missed_pedestrians



    def compute_metrics(self, results: list) -> Dict[str, float]:
        """计算并返回评估指标"""
        total_missed = sum(result["missed_pedestrian_count"] for result in results)
        return {"total_missed_pedestrians": total_missed}


@METRICS.register_module()
class PedestrianDistanceMetricwithline(BaseMetric):
    def __init__(self,
                 pedestrian_idx: int = 11,
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
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pedestrian_idx = pedestrian_idx
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height
        self.camera_height = camera_height
        self.pitch_angle = pitch_angle
        self.known_height = known_height
        self.results = []
        self.image_info = []  # 用于存储每张图像的相关信息

        # 初始化等距线数据列表
        self.equidistant_lines = []  # 每个元素是 (像素高度, 距离)

    def draw_ground_lines(self, image, num_lines: int = 15) -> None:
        """
        在图像上绘制以摄像机为圆心的等距半圆形地面线，并保存等距线数据。
        """
        height, width = image.shape[:2]

        # 相机内参
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        h = self.camera_height  # 相机高度
        theta_pitch = self.pitch_angle  # 相机俯仰角

        # 清空等距线数据
        self.equidistant_lines.clear()

        # 设置实际距离范围
        max_distance = 25  # 最大距离（米）
        min_distance = 1    # 最小距离（米），从 1 米开始
        distances = np.arange(min_distance, max_distance + 1, 1)  # 每隔 1 米
        if num_lines is not None:
            distances = distances[:num_lines]

        for d in distances:
            u_list = []
            v_list = []

            # 角度范围，从 -90 度到 90 度，覆盖摄像机前方的半圆
            for angle_deg in range(-90, 91, 1):
                angle_rad = np.deg2rad(angle_deg)
                # 计算地面点在世界坐标系下的坐标
                X_world = d * np.sin(angle_rad)
                Z_world = d * np.cos(angle_rad)

                # 相机坐标系下的点
                X_cam = X_world
                Y_cam = h  # 相机在地面以上，Y_cam = h
                Z_cam = Z_world

                # 旋转矩阵，考虑俯仰角（绕 X 轴旋转）
                R_pitch = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta_pitch), -np.sin(theta_pitch)],
                    [0, np.sin(theta_pitch), np.cos(theta_pitch)]
                ])

                # 应用旋转
                point_cam = np.array([X_cam, Y_cam, Z_cam])
                point_cam_rotated = R_pitch @ point_cam

                X_c = point_cam_rotated[0]
                Y_c = point_cam_rotated[1]
                Z_c = point_cam_rotated[2]

                if Z_c <= 0:
                    continue  # 忽略相机后方的点

                # 投影到图像坐标系
                u = fx * X_c / Z_c + cx
                v = fy * Y_c / Z_c + cy

                # 检查点是否在图像范围内
                if 0 <= u < width and 0 <= v < height:
                    u_list.append(int(u))
                    v_list.append(int(v))

            # 如果有足够的点，绘制半圆形地面线
            if len(u_list) > 1:
                # 记录当前等距线的平均像素高度（v 值）
                avg_v = int(np.mean(v_list))
                self.equidistant_lines.append((avg_v, d))  # (像素高度, 距离)

                # 绘制等距线
                pts = np.array([[u, v] for u, v in zip(u_list, v_list)], dtype=np.int32)
                cv2.polylines(image, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                # 标注距离
                mid_index = len(u_list) // 2
                cv2.putText(image, f"{d:.1f}m", (u_list[mid_index], v_list[mid_index] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        处理每批数据样本，提取预测和真实标签，计算每个行人的距离。
        """
        for data_sample in data_samples:
            # 获取预测标签
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            image_path = data_sample['img_path']  # 获取图像路径
            image_name = osp.basename(image_path)
            image = cv2.imread(image_path)  # 读取原始图像

            # 绘制等距线并获取等距线数据
            self.draw_ground_lines(image, num_lines=15)
            equidistant_lines = self.equidistant_lines  # 获取等距线数据

            # 提取行人区域
            pred_label_pedestrian = np.where(pred_label == self.pedestrian_idx, 255, 0).astype("uint8")

            # 获取预测结果中的行人轮廓
            ret_pred, thresh_pred = cv2.threshold(pred_label_pedestrian, 127, 255, cv2.THRESH_BINARY)
            pred_contours, hierarchy_pred = cv2.findContours(thresh_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 存储当前图像的行人信息
            pedestrian_info = []
            for contour in pred_contours:
                x, y, w, h = cv2.boundingRect(contour)
                # 获取行人的最低像素点（y + h）
                lowest_point = y + h

                # 比较最低像素点与等距线，确定行人距离
                distance = self.get_distance_from_lines(lowest_point, equidistant_lines)

                # 在图像上绘制行人轮廓和距离信息
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                cv2.putText(image, f'Distance: {distance:.1f}m', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 存储信息
                pedestrian_info.append({
                    'contour': contour,
                    'distance': distance,
                    'bounding_box': (x, y, w, h),
                })

            # 保存带有等距线和行人距离标注的图像
            result_image_name = f'result_{image_name}'
            cv2.imwrite(result_image_name, image)
            print(f'保存带有等距线和行人距离标注的图像: {result_image_name}')

            # 存储图像信息
            self.image_info.append({
                'image_name': image_name,
                'image_path': image_path,
                'pedestrian_info': pedestrian_info,
            })

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

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        计算累计的行人检测指标，例如平均距离。
        """
        # 计算所有行人的平均距离
        all_distances = []
        for info in self.image_info:
            for ped in info['pedestrian_info']:
                all_distances.append(ped['distance'])
        if all_distances:
            average_distance = sum(all_distances) / len(all_distances)
        else:
            average_distance = 0
        print(f'平均行人距离: {average_distance:.2f} 米')
        return {'average_pedestrian_distance': average_distance}


#功能：检测漏检行人，检测已检出行人，距离计算，标出等距线。（揉过的函数）
@METRICS.register_module()
class ComprehensivePedestrianMetric(BaseMetric):
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



#-------------------------------------------------------------0.02<=iou<=0,3---------------------------------------------------#
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

# 移植完成 尚待数据集测试
@METRICS.register_module()
class PedestrianDistanceMetric(BaseMetric):
    def __init__(self,
                 pedestrian_idx: int = 11,
                 rider_idx: int = 12,
                 lane_idx: int = 0,
                 sidewalk_idx: int = 1,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 image_width: int = 2048,
                 image_height: int = 1024,
                 known_height: float = 1.7,  # 行人的实际高度
                 min_area_threshold: int = 100,
                 output_dir: Optional[str] = './pedestrians_detection',
                 min_width: int = 10,
                 min_height: int = 10,
                 max_images_to_save: int = 200,
                 iou_threshold_min: float = 0.02,
                 iou_threshold_max: float = 0.3,
                 fx: float = 2262.52,  # 水平方向焦距
                 fy: float = 2265.30,  # 垂直方向焦距
                 cx: float = 1096.98,  # 图像中心 x 坐标
                 cy: float = 513.137,  # 图像中心 y 坐标
                 camera_height: float = 0.37,  # 相机高度（米）
                 pitch_angle: float = 0.03,  # 相机俯仰角（弧度）
                 max_braking_distance=25.0, # 最大刹车距离
                 **kwargs) -> None:

        # 初始化模型
        super().__init__(collect_device=collect_device, prefix=prefix)

        # 通用参数
        self.pedestrian_idx = pedestrian_idx
        self.rider_idx = rider_idx  # 添加漏掉的 rider_idx
        self.lane_idx = lane_idx
        self.sidewalk_idx = sidewalk_idx
        self.collect_device = collect_device
        self.prefix = prefix

        # 检测参数
        self.min_area_threshold = min_area_threshold
        self.iou_threshold_min = iou_threshold_min
        self.iou_threshold_max = iou_threshold_max
        self.min_width = min_width
        self.min_height = min_height

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

        # 汽车参数
        self.max_braking_distance = max_braking_distance

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
        self.false_pedestrians_dict= {}
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will be used to 
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # 获取预测结果和真实标签
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            
            # 提取图像文件名
            basename = data_sample['img_path']
            if isinstance(basename, (list, tuple)):
                basename = basename[0]
            basename = os.path.basename(basename)
            
            # 提取行人和骑手掩码
            label_pedestrian = self.extract_mask(label, self.pedestrian_idx)
            pred_label_pedestrian = self.extract_mask(pred_label, self.pedestrian_idx)
            
            label_rider = self.extract_mask(label, self.rider_idx)
            pred_label_rider = self.extract_mask(pred_label, self.rider_idx)
            
            # 提取预测的车道和人行道区域
            pred_label_lane = self.extract_mask(pred_label, self.lane_idx)
            pred_label_sidewalk = self.extract_mask(pred_label, self.sidewalk_idx)
            
            # 获取原始图像
            original_img = cv2.imread(data_sample['img_path'])
            
            # 查找漏检和检测到的行人
            missed_pedestrians, detected_pedestrians, false_positives, scoring_results = self.visualize_missed_pedestrians(
                original_img, 
                label_pedestrian,
                pred_label_pedestrian,
                label_rider,
                pred_label_rider,
                pred_label_lane,
                pred_label_sidewalk,
                pred_label
            )
                    # Save images if needed
            if self.output_dir is not None and len(self.saved_image_paths) < self.max_images_to_save:
                self._save_result_images(original_img, basename, 
                                    detected_pedestrians, missed_pedestrians, false_positives)

            # 将结果保存到self.results中
            result = {
                'basename': basename,
                'detected': detected_pedestrians,
                'missed': missed_pedestrians,
                'false_positives': false_positives,
                'scoring_results': scoring_results 
            }
            self.results.append(result)
   
    def visualize_missed_pedestrians(self, img, label_pedestrian, pred_label_pedestrian,
                               label_rider, pred_label_rider,
                               pred_label_lane, pred_label_sidewalk, pred_label):
        """在图像上标出漏检的行人区域，仅在满足 IoU 在 0.02 到 0.3 之间时绘制红框和轮廓"""
        missed_pedestrians = []  # 用于存储 IoU <= 0.3 的行人信息
        detected_pedestrians = []  # 用于存储 IoU > 0.3 的行人信息
        false_positives = []  # 用于存储误检的行人信息

        # 获取轮廓
        gt_contours_pedestrian = self.find_contours(label_pedestrian)
        pred_contours_pedestrian = self.find_contours(pred_label_pedestrian)
        gt_contours_rider = self.find_contours(label_rider)
        pred_contours_rider = self.find_contours(pred_label_rider)

        # 合并轮廓
        gt_contours = gt_contours_pedestrian + gt_contours_rider
        pred_contours = pred_contours_pedestrian + pred_contours_rider

        # 创建IoU矩阵
        iou_matrix = np.zeros((len(gt_contours), len(pred_contours)))
        for i, gt_contour in enumerate(gt_contours):
            for j, pred_contour in enumerate(pred_contours):
                iou_matrix[i, j] = self.compute_iou_contours(gt_contour, pred_contour, label_pedestrian.shape)

        # 使用匈牙利算法进行最优匹配
        matched_pairs = []
        matched_gt_indices = set()
        matched_pred_indices = set()

        # 首先处理高IoU的匹配
        for i in range(len(gt_contours)):
            for j in range(len(pred_contours)):
                if iou_matrix[i, j] > self.iou_threshold_max:
                    if i not in matched_gt_indices and j not in matched_pred_indices:
                        matched_pairs.append((i, j))
                        matched_gt_indices.add(i)
                        matched_pred_indices.add(j)

        # 处理中等IoU的匹配（漏检）
        for i in range(len(gt_contours)):
            if i in matched_gt_indices:
                continue
            best_iou = 0
            best_pred_idx = -1
            for j in range(len(pred_contours)):
                if j in matched_pred_indices:
                    continue
                if self.iou_threshold_min <= iou_matrix[i, j] <= self.iou_threshold_max:
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_pred_idx = j
            if best_pred_idx != -1:
                matched_pairs.append((i, best_pred_idx))
                matched_gt_indices.add(i)
                matched_pred_indices.add(best_pred_idx)

        # 处理匹配结果
        for gt_idx, pred_idx in matched_pairs:
            gt_contour = gt_contours[gt_idx]
            pred_contour = pred_contours[pred_idx]
            iou = iou_matrix[gt_idx, pred_idx]

            # 获取边界框
            x_pred, y_pred, w_pred, h_pred = cv2.boundingRect(pred_contour)

            if iou > self.iou_threshold_max:
                # 处理正确检测
                self._handle_correct_detection(
                    img, x_pred, y_pred, w_pred, h_pred,
                    pred_label, pred_label_lane, pred_label_sidewalk,
                    detected_pedestrians, iou
                )
            else:
                # 处理漏检
                extended_y_bottom, location_label = self.extend_bounding_box(
                    pred_label_lane, pred_label_sidewalk,
                    x_pred, y_pred, w_pred, h_pred, img.shape[0]
                )
                self._handle_missed_detection(
                    img, gt_contour, pred_contour,
                    x_pred, y_pred, w_pred, h_pred,
                    extended_y_bottom, location_label,
                    missed_pedestrians, iou
                )

        # 处理误检（未匹配的预测）
        for pred_idx in range(len(pred_contours)):
            if pred_idx not in matched_pred_indices:
                pred_contour = pred_contours[pred_idx]
                x_pred, y_pred, w_pred, h_pred = cv2.boundingRect(pred_contour)
                self._handle_false_positive(
                    img, pred_contour,
                    x_pred, y_pred, w_pred, h_pred,
                    false_positives
                )

        # 绘制地面线和边界框
        all_pedestrian_boxes = []
        for ped in detected_pedestrians:
            all_pedestrian_boxes.append({'bounding_box': ped['bounding_box'], 'type': 'detected'})
        for ped in missed_pedestrians:
            all_pedestrian_boxes.append({'bounding_box': ped['extended_bounding_box'], 'type': 'missed'})

        self.draw_ground_lines(img, pedestrian_boxes=all_pedestrian_boxes, 
                                detected_pedestrians=detected_pedestrians, 
                                missed_pedestrians=missed_pedestrians)

        # 更新可视化和评分
        scoring_results = self.update_visualization_with_scoring(
            missed_pedestrians,
            detected_pedestrians,
            false_positives,
            img
        )

        cv2.putText(img, f"Total Score: {scoring_results['total_score']:.1f}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return missed_pedestrians, detected_pedestrians, false_positives, scoring_results

    def _handle_correct_detection(self, img, x, y, w, h, pred_label,
                            pred_label_lane, pred_label_sidewalk,
                            detected_pedestrians, iou):
        """处理正确检测的情况，通过分析底部30%区域判断行人位置"""
        # 绘制蓝色框
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 计算底部30%区域的范围
        bottom_start_y = y + int(h * 0.7)  # 从70%处开始
        bottom_height = h - int(h * 0.7)    # 取底部30%

        # 分析底部区域
        bottom_rect = pred_label[bottom_start_y:y + h, x:x + w]
        lane_area = np.sum(bottom_rect == self.lane_idx)
        sidewalk_area = np.sum(bottom_rect == self.sidewalk_idx)

        # 确定位置标签
        location_label = ""
        total_area = lane_area + sidewalk_area
        if total_area > 0:  # 避免除零错误
            lane_ratio = lane_area / total_area
            sidewalk_ratio = sidewalk_area / total_area
            
            # 使用比例阈值判断
            if sidewalk_ratio > 0.6:  # 如果人行道占比超过60%
                location_label = "Sidewalk"
            elif lane_ratio > 0.6:    # 如果车道占比超过60%
                location_label = "Lane"
            else:
                location_label = "Boundary"  # 处于边界位置

        # 添加标签和IoU值
        if location_label:
            text_x = x + w + 5
            text_y = y + h // 2
            cv2.putText(img, location_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(img, f"IoU={iou:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 记录检测信息
        detected_pedestrians.append({
            'bounding_box': (x, y, w, h),
            'iou': float(iou),
            'location_label': location_label,
            'bottom_lane_area': int(lane_area),
            'bottom_sidewalk_area': int(sidewalk_area),
            'bottom_lane_ratio': float(lane_ratio) if total_area > 0 else 0,
            'bottom_sidewalk_ratio': float(sidewalk_ratio) if total_area > 0 else 0
        })

        # Debug: 可视化底部检测区域
        cv2.rectangle(img, (x, bottom_start_y), (x + w, y + h), (0, 255, 0), 1)

    def _handle_missed_detection(self, img, gt_contour, pred_contour,
                                 x, y, w, h, extended_y_bottom, location_label,
                                 missed_pedestrians, iou):
        """处理漏检的情况"""
        # 绘制红色框和轮廓
        cv2.rectangle(img, (x, y), (x + w, extended_y_bottom), (0, 0, 255), 2)
        cv2.drawContours(img, [gt_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [pred_contour], -1, (0, 255, 255), 2)

        # 添加标签
        if location_label:
            text_x = x + w + 5
            text_y = y + h // 2
            cv2.putText(img, location_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(img, f"Missed (IoU={iou:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 记录漏检信息
        missed_pedestrians.append({
            'bounding_box': (x, y, w, h),
            'iou': iou,
            'location_label': location_label,
            'extended_bounding_box': (x, y, w, extended_y_bottom - y)
        })

    def _handle_false_positive(self, img, pred_contour,
                               x, y, w, h, false_positives):
        """处理误检的情况"""
        # 绘制黄色框
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # cv2.putText(img, "False Positive", (x, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 记录误检信息
        false_positives.append({
            'bounding_box': (x, y, w, h)
        })

    def _save_result_images(self, original_img: np.ndarray, basename: str,
                       detected_pedestrians: list, missed_pedestrians: list,
                       false_positives: list) -> None:
        """Helper method to save result images.
        
        Args:
            original_img (np.ndarray): The original image.
            basename (str): Base name of the image file.
            detected_pedestrians (list): List of detected pedestrians.
            missed_pedestrians (list): List of missed pedestrians.
            false_positives (list): List of false positive detections.
        """
        if detected_pedestrians:
            detected_save_path = osp.join(self.output_detected_dir, f"{basename}_detected.jpg")
            cv2.imwrite(detected_save_path, original_img)
            self.saved_image_paths.append(detected_save_path)
            
        if missed_pedestrians:
            missed_save_path = osp.join(self.output_visual_dir, f"{basename}_missed.jpg")
            cv2.imwrite(missed_save_path, original_img)
            self.saved_image_paths.append(missed_save_path)
            
        if false_positives:
            false_save_path = osp.join(self.output_visual_dir, f"{basename}_false.jpg")
            cv2.imwrite(false_save_path, original_img)
            self.saved_image_paths.append(false_save_path)

    def extract_mask(self, label, idx):
        """提取特定索引的二值掩码"""
        return np.where(label == idx, 255, 0).astype("uint8")
    
    def find_contours(self, mask):
        """
        查找轮廓，过滤内部孔洞，并可选保存调试图片。

        Args:
        mask (np.ndarray): 输入的二值化掩码。
        debug_image_path (str, optional): 保存调试图片的路径。

        Returns:
        list: 过滤后的轮廓列表。
        """
        # 二值化处理
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            return []  # 返回空列表而非直接返回原始轮廓

        #获取层级信息
        hierarchy = hierarchy[0]  # [next, previous, first_child, parent]

        #初始化过滤后的轮廓列表
        filtered_contours = []

        #遍历轮廓及其层级
        for i, contour in enumerate(contours):
            # 获取层级信息
            next_contour, prev_contour, first_child, parent = hierarchy[i]

            # 如果轮廓有父级（内部孔洞），跳过
            if parent != -1:
                continue
            filtered_contours.append(contour)
        
        return filtered_contours
    
    def compute_iou_contours(self, contour1, contour2, mask_shape):
        """基于轮廓计算 IoU
        Args:
            contour1: 第一个轮廓
            contour2: 第二个轮廓
            mask_shape: 掩码形状 (height, width)

        Returns:
            float: IoU值
        """
        # 创建空白掩码
        mask1 = np.zeros(mask_shape, dtype=np.uint8)
        mask2 = np.zeros(mask_shape, dtype=np.uint8)

        # 用白色(255)填充轮廓，而不是1
        cv2.drawContours(mask1, [contour1], -1, color=255, thickness=cv2.FILLED)
        cv2.drawContours(mask2, [contour2], -1, color=255, thickness=cv2.FILLED)

        # 转换为二值掩码
        mask1 = mask1 > 0
        mask2 = mask2 > 0

        # 计算交集和并集
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        # 计算IoU
        iou = intersection / union if union > 0 else 0.0

        return iou

    def extend_bounding_box(self, pred_label_lane, pred_label_sidewalk, x_pred, y_pred, w_pred, h_pred, img_height):
        """
        延展预测框到底部，直到与车道或人行道掩码相交。

        Args:
        pred_label_lane (np.ndarray): 车道的二值掩码。
        pred_label_sidewalk (np.ndarray): 人行道的二值掩码。
        x_pred (int): 预测框左上角的 x 坐标。
        y_pred (int): 预测框左上角的 y 坐标。
        w_pred (int): 预测框的宽度。
        h_pred (int): 预测框的高度。
        img_height (int): 图像高度。

        Returns:
        extended_y_bottom (int): 延展后的框底部 y 坐标。
        location_label (str): 延展终止的区域类别（"Lane" 或 "Sidewalk"）。
        """
        lane_mask = (pred_label_lane == 255)  # 车道掩码
        sidewalk_mask = (pred_label_sidewalk == 255)  # 人行道掩码

        # 初始化底部坐标
        extended_y_bottom = y_pred + h_pred
        location_label = ""

        while extended_y_bottom < img_height:
            # 检查当前行是否与车道或人行道掩码相交
            if x_pred + w_pred > lane_mask.shape[1]:
                w_pred = lane_mask.shape[1] - x_pred
            extended_row_lane = lane_mask[extended_y_bottom, x_pred:x_pred + w_pred]
            extended_row_sidewalk = sidewalk_mask[extended_y_bottom, x_pred:x_pred + w_pred]

            lane_intersection = np.any(extended_row_lane)
            sidewalk_intersection = np.any(extended_row_sidewalk)

            if lane_intersection or sidewalk_intersection:
                location_label = "Lane" if lane_intersection else "Sidewalk"
                break

            # 否则继续延展
            extended_y_bottom += 1

        return extended_y_bottom, location_label

    def calculate_contour_score(self, detection_type, iou=None, distance=None):
        """
        Calculate score for each detected/missed contour.

        Args:
            detection_type (str): 'correct', 'missed', or 'false'
            iou (float, optional): Intersection over Union for missed detections

        Returns:
            float: Contour score (0-100)
        """
        if detection_type == 'correct':
            base_score = 100.0  # Full score for correct detection

            if distance is not None and distance <= self.max_braking_distance:
                # Calculate distance ratio (0 to 1)
                distance_ratio = distance / self.max_braking_distance

                # Maximum penalty is 33.33 points, scaled by distance ratio
                distance_penalty = 33.33 * (1 - distance_ratio)

                return max(base_score - distance_penalty, 0)

            return base_score

        if detection_type == 'missed':
            # Penalty is max 33.33 points, scaled by (1-IoU)
            base_penalty = 33.33 * (1 - iou) if iou is not None else 33.33
            # Additional distance penalty for missed detections
            if distance is not None:
                if distance <= self.max_braking_distance:
                    # Calculate distance ratio (0 to 1)
                    distance_ratio = distance / self.max_braking_distance

                    # Add extra penalty based on distance (up to additional 33.33 points)
                    distance_penalty = 33.33 * (1 - distance_ratio)

                    total_penalty = min(base_penalty + distance_penalty, 100)
                    return max(100 - total_penalty, 0)

            return max(100 - base_penalty, 0)

        if detection_type == 'false':
            return 0.0  # Zero points for false positives

        raise ValueError(f"Invalid detection type: {detection_type}")
        
    def update_visualization_with_scoring(self, missed_pedestrians, detected_pedestrians, false_positives, img):
        """
        Update visualization with scoring information

        Args:
        missed_pedestrians (list): List of missed pedestrian detections
        detected_pedestrians (list): List of correct pedestrian detections
        false_positives (list): List of false positive detections
        img (np.ndarray): Input image for visualization

        Returns:
        dict: Scoring results
        """
        scores = []

        # Score correct detections
        for detection in detected_pedestrians:
            distance = detection.get('distance', None)
            score = self.calculate_contour_score('correct', distance=distance)
            scores.append(score)
            x, y, w, h = detection['bounding_box']
            cv2.putText(img, f"Score: {score:.1f}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Score missed detections
        for detection in missed_pedestrians:
            distance = detection.get('distance', None)
            score = self.calculate_contour_score('missed', detection['iou'], distance=distance)
            scores.append(score)
            x, y, w, h = detection['extended_bounding_box']
            cv2.putText(img, f"Score: {score:.1f}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Score false positives
        for detection in false_positives:
            score = self.calculate_contour_score('false')
            # scores.append(score)
            x, y, w, h = detection['bounding_box']
            # cv2.putText(img, f"Score: {score:.1f}", (x, y - 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # Calculate image-level statistics
        total_score = np.mean(scores) if scores else 0

        return {
            'individual_scores': scores,
            'total_score': total_score,
            'detection_counts': {
                'correct': len(detected_pedestrians),
                'missed': len(missed_pedestrians),
                'false_positives': len(false_positives)
            }
        }
    
    def draw_ground_lines(self, img, num_lines: int = 15, pedestrian_boxes: list = None,missed_pedestrians: list = None, detected_pedestrians: list = None) -> None:
        """
        绘制等距线并标注行人边界框与距离。
        """
        height, width = img.shape[:2]

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
                cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                mid_index = len(u_list) // 2
                cv2.putText(img, f"{d:.1f}m", (u_list[mid_index], v_list[mid_index] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 如果传入了行人边界框，标注每个行人的距离
        if pedestrian_boxes:
            for box in pedestrian_boxes:
                x, y, w, h = box['bounding_box']
                lowest_point = y + h  # 获取边界框的最低点
                distance = self.get_distance_from_lines(lowest_point, self.equidistant_lines)

                # 根据行人类型保存距离信息
                if box['type'] == 'detected' and detected_pedestrians:
                    for ped in detected_pedestrians:
                        if np.array_equal(ped['bounding_box'], [x, y, w, h]):
                            ped['distance'] = float(distance)
                            break
                elif box['type'] == 'missed' and missed_pedestrians:
                    for ped in missed_pedestrians:
                        if np.array_equal(ped['extended_bounding_box'], [x, y, w, h]):
                            ped['distance'] = float(distance)
                            break

                # 在图像上绘制距离信息
                cv2.putText(img, f'{distance:.1f}m', (x, y - 10),
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

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute metrics and save all results to JSON files.
        
        Args:
            results (list): The processed results of all batches.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        if self.output_dir is None:
            return {}

        # Initialize a list to store total_scores
        total_scores = []

        # 整理所有结果数据
        missed_pedestrian_dict = {}
        detected_pedestrians_dict = {}
        false_pedestrians_dict = {}
        
        for result in self.results:
            basename = result['basename']
            if result['detected']:
                detected_pedestrians_dict[basename] = result['detected']
            if result['missed']:
                missed_pedestrian_dict[basename] = result['missed']
            if result['false_positives']:
                false_pedestrians_dict[basename] = result['false_positives']
            # Collect total_score from scoring_results
            if 'scoring_results' in result and 'total_score' in result['scoring_results']:
                total_scores.append(result['scoring_results']['total_score'])
        
        # Calculate the average total_score
        if total_scores:
            average_total_score = sum(total_scores) / len(total_scores)
            print(f"Average Total Score: {average_total_score}")
        else:
            average_total_score = 0.0
            print("No total scores found in the results.")
        
        # 保存所有结果到JSON文件
        output_files = {
            "missed_pedestrians.json": missed_pedestrian_dict,
            "detected_pedestrians.json": detected_pedestrians_dict,
            "false_pedestrians.json": false_pedestrians_dict,
            "results.json": {'results': self.results}
        }
        
        for filename, data in output_files.items():
            output_path = osp.join(self.output_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
                
        return {'average_total_score': average_total_score}