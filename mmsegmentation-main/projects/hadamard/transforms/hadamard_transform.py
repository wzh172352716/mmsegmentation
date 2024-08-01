import warnings
from typing import Dict, Optional, Union, Tuple, List

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.datasets import PackSegInputs
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
import reedmuller
import torch

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class HadamardEncodeAnnotations(BaseTransform):

    def __init__(self, ) -> None:

        # create hadamard encoding Look Up Table
        self.rm = reedmuller.reedmuller.ReedMuller(1, 3)
        self.lut = np.zeros((256, 8), dtype=np.uint8)
        self.mapping_dict = {0:1, 1:2, 2:3, 3:4, 8:9, 9:10, 10:11,
                            11:12, 15:14, 16:14, 17:15, 18:15, 255:0}
        for num in range(19):
            mapped_num = self.mapping_dict.get(num, num)
            binary_num = [int(bit) for bit in format(mapped_num, "04b")]
            encoded_codeword = np.array(self.rm.encode(binary_num))
            self.lut[num] = encoded_codeword
        self.lut[255] = np.zeros(8, dtype=np.uint8)

    def hadamard_encode(self, target):
        res = np.take(self.lut, target, axis=0).transpose(2, 0, 1) #/ 4
        return res

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        for key in results.get('seg_fields', []):
            # results[key] = results[key]*0 + 17
            results[key] = self.hadamard_encode(results[key])
            # print("encoded codeword for 13 in hadamard_transform:",results[key][:, 512,512])
        return results


@TRANSFORMS.register_module()
class PackSegInputsHadamard(PackSegInputs):

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
                # print('Unique values in image',np.unique(img),img.shape)
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.float))
                
            else:
                """warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')"""
                data = to_tensor(results['gt_seg_map'].astype(np.float))
            gt_sem_seg_data = dict(data=data)

            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.float)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results