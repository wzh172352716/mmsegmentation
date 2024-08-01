from typing import Optional, List

import torch
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmcv.cnn import Conv2d
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from ..utils import Loss_DC, get_eskd_loss
from mmpretrain.models import BaseClassifier, ImageClassifier
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class ImageClassifierESKD(ImageClassifier):
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
        super(ImageClassifierESKD, self).__init__(backbone, neck, head, pretrained, train_cfg,
                                                data_preprocessor, init_cfg)

        self.lamda_s = lamda_s

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
        x = self.extract_feat(inputs)
        s_logits = self.head(x)
        losses = self.head._get_loss(s_logits, data_samples)

        #losses_teacher = self.head._get_loss(logits_teacher, data_samples)
        #print(losses_teacher)

        dc_loss = get_eskd_loss(self) * self.lamda_s

        losses["loss"] = losses["loss"] + dc_loss
        losses["eskd_term"] = dc_loss
        return losses
