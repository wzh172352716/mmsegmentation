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
from ..utils import Loss_DC


@MODELS.register_module()
class EncoderDecoderKD(EncoderDecoder):

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
                 teacher=None,
                 mask_regularization_reduction='sum',
                 teacher_checkpoint=None,
                 kd_stage_2=True,
                 lamda_s=1,
                 norm_kd=False,
                 kd_loss_indices=[0, 1, 2, 3],
                 kd_loss_shapes=[64, 128, 256, 512, 19],):
        self.mask_regularizer = mask_regularizer
        self.mask_regularization_reduction = mask_regularization_reduction
        super().__init__(backbone=backbone, decode_head=decode_head, neck=neck, auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if isinstance(teacher, dict) and teacher["data_preprocessor"] is None:
            # Merge the data_preprocessor to model config.
            teacher.setdefault('data_preprocessor', data_preprocessor)
        self.teacher = MODELS.build(teacher)

        self.soft_ce_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.lamda_s = lamda_s
        self.kd_stage_2 = kd_stage_2

        if is_model_wrapper(self.teacher):
            self.teacher = self.teacher.module
        else:
            self.teacher = self.teacher
        self.checkpoint = _load_checkpoint(teacher_checkpoint, map_location='cpu')

        self.teacher_init = False
        _load_checkpoint_to_model(self.teacher, self.checkpoint)

        self.teacher = self.teacher.cuda()
        self.correlation_loss = Loss_DC()

        self.norm_kd = norm_kd
        if self.norm_kd:
            self.kd_loss_indices = kd_loss_indices
            self.kd_loss_shapes = kd_loss_shapes
            self.kd_feature_norm = nn.ModuleList([nn.BatchNorm2d(self.kd_loss_shapes[i], affine=False) for i in self.kd_loss_indices])
            self.kd_logits_norm = nn.BatchNorm2d(self.kd_loss_shapes[-1], affine=False)

    def _normalize_features(self, x_student, logits_student):
        norm_x = [bn(fi) for bn, fi in zip(self.kd_feature_norm, x_student)]
        norm_logits = self.kd_logits_norm(logits_student)

        return norm_x, norm_logits

    def _get_teacher_predictions(self, inputs: Tensor):
        if not self.teacher_init:
            self.teacher_init = True
            _load_checkpoint_to_model(self.teacher, self.checkpoint)

        with torch.no_grad():
            x_teacher = self.teacher.extract_feat(inputs)
            logits_teacher = self.teacher.decode_head.forward(x_teacher)

        return x_teacher, logits_teacher

    def _get_knowledge_distillation_losses(self, x_student, x_teacher, logits_student, logits_teacher):
        if self.norm_kd:
            x_student, logits_student = self._normalize_features(x_student, logits_student)
        # stage_1_kd_loss = 0
        # for xi_student, xi_teacher in zip(x_student, x_teacher):
        #    stage_1_kd_loss = stage_1_kd_loss + self.mse_loss(xi_student, xi_teacher)

        # stage_2_kd_loss = self.soft_ce_loss(logits_student, F.softmax(logits_teacher, dim=1)) * self.lamda_s
        correlation_loss = self.correlation_loss(x_teacher, x_student) * self.lamda_s
        correlation_loss = correlation_loss + self.correlation_loss(logits_student, logits_teacher) * self.lamda_s

        return correlation_loss

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)

        losses = dict()
        seg_logits = self.decode_head.forward(x)
        loss_decode = self.decode_head.loss_by_feat(seg_logits, data_samples)

        losses.update(add_prefix(loss_decode, 'decode'))
        losses.update(loss_decode)

        losses["decode.loss_ce"] = losses["decode.loss_ce"]

        x_teacher, logits_teacher = self._get_teacher_predictions(inputs)

        losses["teacher.acc"] = self.teacher.decode_head.loss_by_feat(logits_teacher, data_samples)["acc_seg"]

        dc_loss = self._get_knowledge_distillation_losses(x, x_teacher, seg_logits, logits_teacher)
        losses["decode.dc_loss"] = dc_loss

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
