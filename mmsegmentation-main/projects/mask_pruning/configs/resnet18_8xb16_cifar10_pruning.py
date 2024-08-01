_base_ = [
    'mmpretrain::_base_/models/resnet18_cifar.py', '../../classification/configs/cifar10_bs16.py',
    'mmpretrain::_base_/schedules/cifar10_bs128.py', 'mmpretrain::_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifierPruning',
    backbone=dict(
        conv_cfg=dict(type='Conv2dPruning')),
    )

custom_hooks = [
    dict(type='MaskPruningHook', do_explicit_pruning=False, prune_at_start=False, logging_interval=500, pruning_interval=5000, debug=True),
]