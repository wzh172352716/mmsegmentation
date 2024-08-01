# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from reedmuller import reedmuller
from torch import Tensor

from mmseg.models import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from mmengine.registry import TRANSFORMS

import time


@MODELS.register_module()
class EncoderDecoderDistort(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 distort_cfg: OptMultiConfig = None,
                 resize=None,
                 resize_after=None):
        super().__init__(backbone=backbone, decode_head=decode_head, neck=neck, auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg, pretrained=pretrained,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.distort = TRANSFORMS.build(distort_cfg)
        self.resize = resize
        self.resize_after = resize_after

    def decode_labels(self, labels):
        decimal_labels = labels * self.binary_to_decimal_tensor.to(labels.device).unsqueeze(1).unsqueeze(1)
        decimal_labels = decimal_labels.sum(dim=0).long()
        res = torch.take(self.decoding_lut.to(labels.device), decimal_labels)
        return res

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom = \
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                               padding_top:H - padding_bottom,
                               padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3,))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2,))

                # resize as original shape
                if self.resize is not None:
                    i_seg_logits = resize(
                        i_seg_logits,
                        size=self.resize,
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False)
                """print(i_seg_logits.shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
                print(i_seg_logits.shape)"""
                i_seg_logits = i_seg_logits.squeeze(0)
                #print(i_seg_logits.shape)
                device = i_seg_logits.device
                img = i_seg_logits.permute(1, 2, 0).cpu().numpy()
                img = self.distort.distort_reverse(img)
                i_seg_logits = torch.tensor(img, device=device).permute(2, 0, 1)
                #print(i_seg_logits.shape)
                if self.resize_after is not None:
                    i_seg_logits = resize(
                        i_seg_logits,
                        size=self.resize_after,
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                    PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                    PixelData(**{'data': i_seg_pred})
            })

        return data_samples
