from typing import Optional, Sequence, Dict
from mmengine.evaluator import BaseMetric  # 从 MMEngine 导入基础 Metric 类
import torch
import torch.nn.functional as F  # 用于处理张量和神经网络功能的 PyTorch 模块
from mmseg.registry import METRICS  # 从 MMSegmentation 导入注册器，用于注册自定义 Metric


@METRICS.register_module()  # 将这个类注册为 MMSegmentation 的度量工具模块
class AccuracyMetric(BaseMetric):
    def __init__(self,
                 accuracyD=True,  # 是否计算直接精度
                 accuracyI=True,  # 是否计算图像级别精度
                 accuracyC=True,  # 是否计算类别级别精度
                 q=1,  # 用于精度计算的百分比参数
                 binary=False,  # 是否为二分类任务
                 num_classes=19,  # 类别数量，默认为19
                 ignore_index=None,  # 忽略标签中的某些索引
                 collect_device: str = 'cpu',  # 计算时使用的设备，默认为'cpu'
                 prefix: Optional[str] = None):  # 度量名称的前缀
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.accuracyD = accuracyD
        self.accuracyI = accuracyI
        self.accuracyC = accuracyC
        self.q = q
        self.binary = binary
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tp = torch.tensor([])  # 真阳性
        self.tn = torch.tensor([])  # 真阴性
        self.fp = torch.tensor([])  # 假阳性
        self.fn = torch.tensor([])  # 假阴性
        self.active_classes = torch.tensor([], dtype=torch.bool)  # 活跃类的布尔掩码
        self.image_file = []  # 存储图像文件名
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_sem_seg']['data'].squeeze().cpu()  # 获取预测值并移到CPU上
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu()  # 获取真实标签
            image_file = data_sample.get('img_path', None)  # 获取图像文件名
            batch_size = pred.size(0)
            pred = pred.view(batch_size, -1)
            label = label.view(batch_size, -1)
            keep_mask = (label != self.ignore_index)  # 忽略指定的索引
            keep_mask = keep_mask.unsqueeze(1).expand(batch_size, self.num_classes, -1)
            pred = F.one_hot(pred, num_classes=self.num_classes).permute(0, 2, 1)  # 独热编码预测结果
            label = torch.clamp(label, 0, self.num_classes - 1)  # 将标签值限制在类别范围内
            label = F.one_hot(label, num_classes=self.num_classes).permute(0, 2, 1)
            for i in range(batch_size):
                keep_mask_i = keep_mask[i, :, :]
                pred_i = pred[i, :, :][keep_mask_i].reshape(self.num_classes, -1)
                label_i = label[i, :, :][keep_mask_i].reshape(self.num_classes, -1)

                if label_i.size(1) < 1:
                    continue
                tp = torch.logical_and(pred_i == 1, label_i == 1)  # 真阳性
                tn = torch.logical_and(pred_i == 0, label_i == 0)  # 真阴性
                fp = torch.logical_and(pred_i == 1, label_i == 0)  # 假阳性
                fn = torch.logical_and(pred_i == 0, label_i == 1)  # 假阴性
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
                self.image_file.append(image_file_i)
    def compute_metrics(self, results: list) -> Dict[str, float]:
        final_results = {}
        if self.accuracyD:
            final_results.update(self.valueD())
        if self.accuracyI:
            final_results.update(self.valueI())
        if self.accuracyC:
            final_results.update(self.valueC())
        return final_results
