from typing import Optional, Sequence, Dict

from mmengine.evaluator import BaseMetric

import numpy as np
import cv2
from mmseg.registry import METRICS
import collections
import os.path as osp


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

            #cv2.imwrite("debug_label.png", label_pedestrian)

            ret, thresh = cv2.threshold(label_pedestrian, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            res = self.get_predicted_contours(contours, pred_label_pedestrian)

            for key, value in res.items():
                if key >= 200 and value["total_num"] > value["recognized_num"]:
                    basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
                    print(f"For image {basename} an pedestrian of large size (>=200px) was not predicted!")

                    out = cv2.cvtColor(label_pedestrian,cv2.COLOR_GRAY2RGB)
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
                #final_result[key]["contours"].extend(value["contours"])
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
        #pred_mask_binary = np.where(pred_mask == 11, 255, 0).astype("uint8")

        for contour, height in zip(contours, heights):
            contour_img = np.zeros_like(pred_mask)
            cv2.drawContours(contour_img, [contour], -1, color=(255), thickness=cv2.FILLED)

            if not height in res:
                res[height] = {"total_num": 0, "recognized_num": 0, "contours": []}
            res[height]["total_num"] += 1
            intersection = cv2.bitwise_and(contour_img, pred_mask)
            #cv2.imwrite("intersection.png", intersection)
            if np.max(intersection) > 0:
                res[height]["recognized_num"] += 1
                res[height]["contours"].append(contour)

        return res