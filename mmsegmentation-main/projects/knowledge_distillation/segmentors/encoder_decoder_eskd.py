# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional

import mmcv
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model, load_state_dict
from torch import Tensor

from mmseg.models import accuracy
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor
from mmcv.cnn import Conv2d
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from ..utils import Loss_DC, get_eskd_loss


@MODELS.register_module()
class EncoderDecoderESKD(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 mask_regularizer=0.005,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 mask_regularization_reduction='sum',
                 lamda_s=1):
        self.mask_regularizer = mask_regularizer
        self.mask_regularization_reduction = mask_regularization_reduction
        super().__init__(backbone=backbone, decode_head=decode_head, neck=neck, auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)


        self.soft_ce_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.lamda_s = lamda_s


        self.correlation_loss = Loss_DC()


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)

        losses = dict()
        seg_logits = self.decode_head.forward(x)
        loss_decode = self.decode_head.loss_by_feat(seg_logits, data_samples)

        losses.update(add_prefix(loss_decode, 'decode'))
        losses.update(loss_decode)

        losses["decode.loss_ce"] = losses["decode.loss_ce"]
        losses["decode.eskd_loss"] = get_eskd_loss(self) * self.lamda_s

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
