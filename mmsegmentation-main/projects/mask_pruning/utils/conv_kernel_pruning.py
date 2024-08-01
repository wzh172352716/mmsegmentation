import torch
from mmengine.registry import MODELS
from torch import nn

from .mask_wrapper import mask_class_wrapper
from .identity_conv import IdentityConv2d, EmptyModule

Conv2dKernelPruning = mask_class_wrapper(nn.Conv2d, mode="conv_kernel")
MODELS.register_module('Conv2dKernelPruning', module=Conv2dKernelPruning)

Conv2dPruning = mask_class_wrapper(nn.Conv2d, mode="conv")
MODELS.register_module('Conv2dPruning', module=Conv2dPruning)


class IdentityConv2dPruned(IdentityConv2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        dim=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype, Conv2dPruning, dim)
        #for n, p in self.conv.named_modules():
        #    setattr(self, 'n', p)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.module_list = self.conv.module_list

    def forward(self, x):
        if isinstance(self.conv, EmptyModule):
            b, c, h, w = x.shape
            return torch.zeros((b, self.out_channels, h, h), requires_grad=True, device=x.device)
        return super().forward(x)


MODELS.register_module('Conv2dPruningIdent', module=IdentityConv2dPruned)

