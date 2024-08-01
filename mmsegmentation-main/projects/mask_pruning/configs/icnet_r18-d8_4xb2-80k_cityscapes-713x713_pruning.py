_base_ = [
    '../../../configs/_base_/models/icnet_r50-d8.py',
    '../../../configs/_base_/datasets/cityscapes_832x832.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_80k.py'
]
crop_size = (713, 713)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
#crop_size = (832, 832)
model = dict(
    type="PrunedEncoderDecoder", 
    mask_factor=0.1,
    backbone=dict(layer_channels=(128, 512), backbone_cfg=dict(depth=18),conv_cfg=dict(type='Conv2dPruning')))

optim_wrapper = dict(
    #_delete_=True,
    #type='OptimWrapper',
    #optimizer=dict(
        #type='AdamW', lr=0.000015, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
           # 'pos_block': dict(decay_mult=0.),
            #'norm': dict(decay_mult=0.),
            #'head': dict(lr_mult=10.),
            'p1': dict(lr_mult=10., weight_decay=0),
            #'p1': dict(lr=CosineDecay(0.01, T_max=10),lr_mult=10., weight_decay=0)
        }))

custom_hooks = [
    dict(type='MaskPruningHook', do_explicit_pruning=False, prune_at_start=False, logging_interval=5000, pruning_interval=5000, debug=False),
]
