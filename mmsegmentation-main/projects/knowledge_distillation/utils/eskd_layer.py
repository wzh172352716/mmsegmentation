import torch
from torch import Tensor
from mmengine.registry import MODELS

from .Partial_DC_grad import Loss_DC
import torch.nn.functional as F


def get_eskd_loss(model):
    sum = 0
    for n, p in model.named_modules():
        if isinstance(p, ESKDConv2d) or isinstance(p, ESKDLinear):
            sum += p.get_eskd_loss()
    return sum


@MODELS.register_module()
class ESKDConv2d(torch.nn.Conv2d):

    def get_eskd_loss(self):
        return getattr(self, "eskd_loss", 0)

    def forward(self, input: Tensor) -> Tensor:
        if not hasattr(self, "correlation_loss"):
            self.correlation_loss = Loss_DC()
        res = self._conv_forward(input, self.weight, self.bias)
        copy = torch.tensor(res, requires_grad=False)
        self.eskd_loss = self.correlation_loss(input, copy)
        return res


@MODELS.register_module()
class ESKDLinear(torch.nn.Linear):

    def get_eskd_loss(self):
        return getattr(self, "eskd_loss", 0)

    def forward(self, input: Tensor) -> Tensor:
        if not hasattr(self, "correlation_loss"):
            self.correlation_loss = Loss_DC()
        res = F.linear(input, self.weight, self.bias)
        copy = torch.tensor(res, requires_grad=False)
        self.eskd_loss = self.correlation_loss(input, copy)
        return res
