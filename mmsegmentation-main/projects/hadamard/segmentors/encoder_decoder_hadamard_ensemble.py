# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from reedmuller import reedmuller
from torch import Tensor

from mmseg.models import EncoderDecoder, BaseSegmentor
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder_hadamard import EncoderDecoderHadamard
import numpy as np
import random

import time


@MODELS.register_module()
class EncoderDecoderHadamardEnsemble(EncoderDecoderHadamard):

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
                 init_cfg: OptMultiConfig = None,
                 models = []):

        super().__init__(backbone=backbone, decode_head=decode_head, threshold=threshold, neck=neck, auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg, pretrained=pretrained,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.models = models
        self.data_preprocessor = self.models[0].data_preprocessor
        self.method = "add"

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        logits_list = []
        for m in self.models:
            logits = m.inference(inputs, batch_img_metas)
            logits_list.append(logits)
        #return logits_list[0]

        sum_logits = 0
        if self.method == "add":
            for l in logits_list:
                sum_logits = sum_logits + l
            sum_logits = sum_logits / len(logits_list)

        return sum_logits