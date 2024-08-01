_base_ = [
    '../../../configs/_base_/models/pspnet_r50-d8.py', '../../../configs/_base_/datasets/cityscapes.py',
    '../../../configs/_base_/default_runtime.py', '../../../configs/_base_/schedules/schedule_80k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(type="PrunedEncoderDecoder",
             data_preprocessor=data_preprocessor,
             backbone=dict(
                 conv_cfg=dict(type='Conv2dPruning')
             ),
             )

custom_hooks = [
    dict(type='MaskPruningHook', do_explicit_pruning=False, prune_at_start=False, logging_interval=5000, pruning_interval=5000, debug=False),
]