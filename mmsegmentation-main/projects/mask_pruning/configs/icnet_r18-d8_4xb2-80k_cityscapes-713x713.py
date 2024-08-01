_base_ = './icnet_r50-d8_4xb2-80k_cityscapes-713x713.py'
#crop_size = (832, 832)
model = dict(
    backbone=dict(layer_channels=(128, 512), backbone_cfg=dict(depth=18)))
