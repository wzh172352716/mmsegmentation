from collections import namedtuple

import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Distortion(BaseTransform):


    def __init__(self, size=(2048, 1024), compression_rate=1.0, labels=True, max_distortion=0):
        self.compression_rate = compression_rate
        self.size = size
        self.labels = labels
        self.max_distortion = max_distortion
        self.trans_x, self.trans_y, self.trans_x_back, self.trans_y_back,\
            self.min_y, self.max_y = self.create_args(size[0], size[1])
        #print(f"Distortion Output size: {self.max_y - self.min_y}")

    def create_args(self, width, height):
        Point = namedtuple('Point', ['x', 'y', 'i', 'j'])
        pixel_size = height / self.compression_rate
        center_y = int(height // 2.5)
        min_y = height
        max_y = 0

        flex_x = np.zeros((height, width), np.float32)
        flex_y = np.zeros((height, width), np.float32)
        flex_x_back = np.zeros((height, width), np.float32)
        flex_y_back = np.zeros((height, width), np.float32)

        for x in range(width):
            for y in range(height):
                diff_center_y = center_y - y
                if diff_center_y > 0:
                    p = Point(x, y, x, int(center_y - max(self.max_distortion, 1 / ((1 / pixel_size) * diff_center_y + 1)) * diff_center_y))
                else:
                    p = Point(x, y, x,
                              int(center_y + max(self.max_distortion, 1 / ((1 / pixel_size) * abs(diff_center_y) + 1)) * abs(diff_center_y)))
                # print(p)
                flex_x[p[3], x] = x  # p1[0]
                flex_y[p[3], x] = y

                if diff_center_y > 0:
                    min_y = min(min_y, p[3])
                else:
                    max_y = max(max_y, p[3])

                flex_x_back[y, x] = x
                flex_y_back[y, x] = p[3]

        return flex_x, flex_y, flex_x_back, flex_y_back, min_y, max_y

    def distort(self, img):
        return cv2.remap(img, self.trans_x, self.trans_y, cv2.INTER_LINEAR)[self.min_y:self.max_y+1]

    def distort_reverse(self, img):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        z = np.zeros((self.size[1], self.size[0], img.shape[-1]))
        z[self.min_y:self.max_y+1] = img
        z = cv2.remap(z, self.trans_x_back, self.trans_y_back, cv2.INTER_LINEAR)
        return z
    def transform(self, results: dict) -> dict:
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = self.distort(results['img'])
        results['img_shape'] = results['img'].shape[:2]
        if self.labels:
            for key in results.get('seg_fields', []):
                results[key] = self.distort(results[key])
        #print(results['img'].shape[:2])
        return results

