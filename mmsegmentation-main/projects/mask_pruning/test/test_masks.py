import pytest
import sys
import torch

from mmengine import print_log

from ..utils.masks import LearnableKernelMask

import logging


def test_learnable_kernel_mask():
    def get_mask(mask):
        res = mask.get_mask()[0]
        res = torch.where(res < 0.001, 0, res)
        res = torch.where(torch.abs(res - 1) < 0.001, 1, res)
        return res

    mask = LearnableKernelMask(1, 1, 5, lr_mult_factor=1, k=20)

    kernel_mask = get_mask(mask)
    target_mask = torch.ones(1, 1, 5, 5)
    assert torch.all(torch.eq(kernel_mask, target_mask))

    mask.p1 = torch.nn.Parameter(mask.p1 + 1.5)
    kernel_mask = get_mask(mask)
    target_mask[0, 0, :, 4] = 0.0
    target_mask[0, 0, 4, :] = 0.0
    assert torch.all(torch.eq(kernel_mask, target_mask))

    mask.p1 = torch.nn.Parameter(mask.p1 + torch.tensor([0, 1]))
    kernel_mask = get_mask(mask)
    target_mask[0, 0, :, 0] = 0.0
    assert torch.all(torch.eq(kernel_mask, target_mask))

    mask.p1 = torch.nn.Parameter(mask.p1 + torch.tensor([0, 1]))
    kernel_mask = get_mask(mask)
    target_mask[0, 0, :, 3] = 0.0
    assert torch.all(torch.eq(kernel_mask, target_mask))

    mask.p1 = torch.nn.Parameter(mask.p1 + torch.tensor([1, 0]))
    kernel_mask = get_mask(mask)
    target_mask[0, 0, 0, :] = 0.0
    assert torch.all(torch.eq(kernel_mask, target_mask))

    mask.p1 = torch.nn.Parameter(mask.p1 + torch.tensor([0, 1]))
    kernel_mask = get_mask(mask)
    target_mask[0, 0, :, 1] = 0.0
    assert torch.all(torch.eq(kernel_mask, target_mask))

def test_learnable_kernel_mask_sizes():
    mask = LearnableKernelMask(1, 10, 5, lr_mult_factor=1, k=20)

    mask.p1 = torch.nn.Parameter(mask.p1 + 1)
    mask.p1 = torch.nn.Parameter(mask.p1 + torch.tensor([kx1, kx2, kx1, kx2]))

"""class TestMasks(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)"""
