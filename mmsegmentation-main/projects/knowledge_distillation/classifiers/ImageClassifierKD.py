from typing import Optional, List

import torch
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmcv.cnn import Conv2d
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from ..utils import Loss_DC
from mmpretrain.models import BaseClassifier, ImageClassifier
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class ImageClassifierKD(ImageClassifier):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 teacher=None,
                 teacher_checkpoint=None,
                 lamda_s=1,
                 logits_kd=True,
                 kd_feature_indices=None):
        super(ImageClassifierKD, self).__init__(backbone, neck, head, pretrained, train_cfg,
                                                data_preprocessor, init_cfg)
        #if isinstance(teacher, dict) and teacher["data_preprocessor"] is None:
            # Merge the data_preprocessor to model config.
            #teacher.setdefault('data_preprocessor', data_preprocessor)
        self.teacher = MODELS.build(teacher)
        self.lamda_s = lamda_s

        if is_model_wrapper(self.teacher):
            self.teacher = self.teacher.module
        else:
            self.teacher = self.teacher
        self.checkpoint = _load_checkpoint(teacher_checkpoint, map_location='cpu')

        self.teacher_init = False
        _load_checkpoint_to_model(self.teacher, self.checkpoint)

        self.teacher = self.teacher.cuda()
        self.correlation_loss = Loss_DC()
        self.logits_kd = logits_kd
        self.kd_feature_indices = kd_feature_indices

    def _get_teacher_predictions(self, inputs: Tensor):
        if not self.teacher_init:
            self.teacher_init = True
            _load_checkpoint_to_model(self.teacher, self.checkpoint)

        with torch.no_grad():
            x_teacher = self.teacher.extract_feat(inputs)
            logits_teacher = self.teacher.head(x_teacher)

        return x_teacher, logits_teacher

    def _get_knowledge_distillation_losses(self, x_student, x_teacher, logits_student, logits_teacher):
        # stage_1_kd_loss = 0
        # for xi_student, xi_teacher in zip(x_student, x_teacher):
        #    stage_1_kd_loss = stage_1_kd_loss + self.mse_loss(xi_student, xi_teacher)

        # stage_2_kd_loss = self.soft_ce_loss(logits_student, F.softmax(logits_teacher, dim=1)) * self.lamda_s
        if self.logits_kd:
            if self.kd_feature_indices is None:
                correlation_loss = (len(x_teacher) + 1 + self.correlation_loss(x_teacher, x_student)) * self.lamda_s
            else:
                correlation_loss = (len(self.kd_feature_indices) + 1 + self.correlation_loss([x_teacher[i] for i in self.kd_feature_indices], [x_student[i] for i in self.kd_feature_indices])) * self.lamda_s
        else:
            correlation_loss = self.lamda_s
        correlation_loss = correlation_loss + self.correlation_loss(logits_student, logits_teacher) * self.lamda_s

        return correlation_loss

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #torch.autograd.set_detect_anomaly(True)
        x = self.extract_feat(inputs)
        s_logits = self.head(x)
        losses = self.head._get_loss(s_logits, data_samples)

        x_teacher, logits_teacher = self._get_teacher_predictions(inputs)
        #losses_teacher = self.head._get_loss(logits_teacher, data_samples)
        #print(losses_teacher)

        dc_loss = self._get_knowledge_distillation_losses(x, x_teacher, s_logits, logits_teacher)

        losses["loss"] = losses["loss"] + dc_loss
        losses["kd_term"] = dc_loss
        return losses
