_base_ = [
    'mmpretrain::_base_/models/mobilenet_v2_1x.py', 'mmpretrain::_base_/datasets/cifar100_bs16.py',
    'mmpretrain::_base_/schedules/cifar10_bs128.py', 'mmpretrain::_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005))

train_cfg = dict(by_epoch=True, max_epochs=240, val_interval=10)
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[150, 180, 210],
    gamma=0.1,
)

train_dataloader = dict(batch_size=64,)

