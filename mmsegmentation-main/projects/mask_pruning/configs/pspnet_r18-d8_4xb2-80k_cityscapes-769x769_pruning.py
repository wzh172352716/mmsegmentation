_base_ = '../../../configs/pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-769x769.py'
model = dict(
    type='PrunedEncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18, conv_cfg=dict(type='Conv2dPruning')),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
