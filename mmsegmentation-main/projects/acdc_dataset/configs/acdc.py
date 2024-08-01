_base_ = [
    '../../../configs/_base_/datasets/cityscapes_1024x1024.py',
]

# dataset settings
dataset_type = 'ACDCDataset'
data_root = 'data/acdc/'


train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/val', seg_map_path='gt/val')
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/val', seg_map_path='gt/val')
    ))
test_dataloader = val_dataloader