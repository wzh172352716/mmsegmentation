_base_ = [
    '../../../configs/_base_/models/segformer_mit-b0.py',
    '../../../configs/_base_/datasets/cityscapes_1024x1024.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_160k.py'
]
compressed_height = 677
compression_rate = 1
crop_size = (1024, compressed_height)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

#1 -> 677
#2 -> 508
#3 -> 407
#4 -> 339
#5 -> 291
#6 -> 255
#7 -> 227
#8 -> 204
#9 -> 186
#10 -> 170
distort_cfg = dict(type='Distortion', size=(2048, 1024), compression_rate=compression_rate, labels=False)
model = dict(
    type="EncoderDecoderDistort",
    distort_cfg=distort_cfg,
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(compressed_height, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=3000,
        end=320000,
        by_epoch=False,
    )
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Distortion', size=(2048, 1024), compression_rate=compression_rate, labels=True),
    dict(
        type='RandomResize',
        scale=(2048, compressed_height),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    distort_cfg,
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000*2, val_interval=16000*2)
train_dataloader = dict(batch_size=1, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

