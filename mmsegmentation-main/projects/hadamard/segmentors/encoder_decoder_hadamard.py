# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from reedmuller import reedmuller
from torch import Tensor
import numpy as np
import random

from mmseg.models import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

# import time

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

@MODELS.register_module()
class EncoderDecoderHadamard(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 threshold: int,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(backbone=backbone, decode_head=decode_head, neck=neck, auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg, pretrained=pretrained,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.rm = reedmuller.ReedMuller(1, 3)
        self.binary_to_decimal_tensor = torch.tensor([2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 2**0])
        self.decoding_lut = torch.zeros(256, dtype=torch.uint8)
        self.threshold = threshold

        print(self.threshold)

        for i in range(256):
            word = self.rm.decode([int(bit) for bit in format(i, "08b")])
            if word is None:
                failed_word = [int(bit) for bit in format(i, "08b")]
                # print(failed_word)
                # lut_codeword = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                lut_codeword = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                                        [0, 0, 1, 1, 0, 0, 1, 1],
                                        [0, 0, 1, 1, 1, 1, 0, 0],
                                        [0, 1, 0, 1, 0, 1, 0, 1],
                                        [0, 1, 0, 1, 1, 0, 1, 0],
                                        [0, 1, 1, 0, 0, 1, 1, 0],
                                        [0, 1, 1, 0, 1, 0, 0, 1],
                #                          [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 1, 1, 0, 0],
                                        [1, 1, 0, 0, 0, 0, 1, 1], # sky
                                        [1, 0, 1, 0, 1, 0, 1, 0],
                                        [1, 0, 1, 0, 0, 1, 0, 1],
                                        [1, 0, 0, 1, 1, 0, 0, 1],
                                        [1, 0, 0, 1, 0, 1, 1, 0]])
                
                min_distance = float('inf')
                equidistant_codewords = []    
                received_codeword = failed_word
                for i in range(len(lut_codeword)):
                    current_distance = np.sum(np.bitwise_xor(received_codeword, lut_codeword[i]))
                    if current_distance < min_distance:
                        min_distance = current_distance
                        equidistant_codewords = [i]
                    elif current_distance == min_distance:
                        equidistant_codewords.append(i)

                selected_codeword_index = random.choice(equidistant_codewords)
                # print(selected_codeword_index)
                word = lut_codeword[selected_codeword_index]

            # if word is None:
            #     word = torch.tensor([0])
                
            else:
                num = torch.tensor([int(''.join(map(str, word)), 2)])
            self.decoding_lut[i] = num

    def decode_labels(self, labels):
        decimal_labels = labels * self.binary_to_decimal_tensor.to(labels.device).unsqueeze(1).unsqueeze(1)
        decimal_labels = decimal_labels.sum(dim=0).long()
        res = torch.take(self.decoding_lut.to(labels.device), decimal_labels)
        # mapping_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:255, 9:7,
        #                 10:8 ,11:9, 12:10, 13:11, 14:12, 15:13, 0:255}
        mapping_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 9:7, 10:8,
                            11:9, 12:10, 13:11, 14:12, 15:13, 0:255}
        mapping_function = np.vectorize(lambda x: mapping_dict.get(x, 0))

        # Move the tensor to CPU before converting to NumPy
        res_cpu = res.cpu().numpy()
        res_cpu = mapping_function(res_cpu)
        res = torch.tensor(res_cpu)
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

            i_seg_logits = i_seg_logits.softmax(dim = 0)

            i_seg_pred = torch.where(i_seg_logits >=self.threshold, torch.tensor(1.0), torch.tensor(0.0))

            gt_sem_seg = data_samples[i].gt_sem_seg.data
            
            # print("encoded codeword for 13 in encoder_decoder_hadamard:",gt_sem_seg[:, 512,512])
            gt_sem_seg = self.decode_labels(gt_sem_seg) #* 4)
            
            # print("------------------------------------------------------")
            # print("Unique values in label tensor after decoding:",torch.unique(gt_sem_seg),gt_sem_seg.shape)
            # print("------------------------------------------------------")

            i_seg_pred = self.decode_labels(i_seg_pred)

            # print("------------------------------------------------------")
            # print("Unique values in pred tensor after decoding:",torch.unique(i_seg_pred),i_seg_pred.shape)
            # print("------------------------------------------------------")

            data_samples[i].set_data({
                'seg_logits':
                    PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                    PixelData(**{'data': i_seg_pred}),
                'gt_sem_seg':
                    PixelData(**{'data': gt_sem_seg}),
            })

        return data_samples
