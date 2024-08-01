# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityscapesDatasetFastRestart(CityscapesDataset):

    def __getitem__(self, idx: int) -> dict:
        if getattr(self, 'fast_forward', False):
            print("-")
            return dict()
        else:
            print("+")
            return super().__getitem__(idx)
