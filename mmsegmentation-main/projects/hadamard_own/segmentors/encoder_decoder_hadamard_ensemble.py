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
from .encoder_decoder_hadamard import EncoderDecoderHadamard

import time


@MODELS.register_module()
class EncoderDecoderHadamardEnsemble(EncoderDecoderHadamard):

    def __init__(self,
                 models,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):

        EncoderDecoder.__init__(self,
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.models = models
        self.method = "add"

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.rm = reedmuller.ReedMuller(1, 3)
        self.binary_to_decimal_tensor = torch.tensor([2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4, 2 ** 3, 2 ** 2, 2 ** 1, 2 ** 0])
        self.decoding_lut = torch.zeros(256, dtype=torch.uint8)
        for i in range(256):
            word = self.rm.decode([int(bit) for bit in format(i, "08b")])
            if word is None:
                word = torch.tensor([0])
            else:
                word = torch.tensor([int(''.join(map(str, word)), 2)])

            num = word - 1
            if num == 15:
                num = 19
            elif num == 12:
                num = 18
            elif num == 10:
                num = 17
            elif num == -1:
                num = 255

            self.decoding_lut[i] = word

    def _encode_decode_complete_ensemble(self, inputs, batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        logits_list = []

        for m in self.models:
            feature = m.extract_feat(inputs)
            logits = m.decode_head.predict(feature, batch_img_metas,
                                              self.test_cfg)
            logits_list.append(logits)

        if self.method == "add":
            sum_logits = 0
            for l in logits_list:
                sum_logits = sum_logits + l
            sum_logits = sum_logits / len(logits_list)

        return sum_logits

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]):
        return self._encode_decode_complete_ensemble(inputs, batch_img_metas)

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
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            # print(i_seg_pred.shape)
            i_seg_logits = i_seg_logits.sigmoid()
            i_seg_pred = torch.round(i_seg_logits)

            gt_sem_seg = data_samples[i].gt_sem_seg.data

            gt_sem_seg = self.decode_labels(gt_sem_seg)
            i_seg_pred = self.decode_labels(i_seg_pred)

            data_samples[i].set_data({
                'seg_logits':
                    PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                    PixelData(**{'data': i_seg_pred}),
                'gt_sem_seg':
                    PixelData(**{'data': gt_sem_seg}),
            })

        return data_samples
