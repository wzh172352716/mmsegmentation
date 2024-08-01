_base_ = [
    'mmpretrain::_base_/models/resnet18_cifar.py', './cifar10_bs16.py',
    'mmpretrain::_base_/schedules/cifar10_bs128.py', 'mmpretrain::_base_/default_runtime.py'
]