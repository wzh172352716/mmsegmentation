_base_ = './deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))

# custom_imports = dict(
#     imports=['/home/teamproject_mmseg/work_fast/mmsegmentation-main/mmseg/evaluation/metrics'],  # 这里引用模块路径
#     allow_failed_imports=False
# )
