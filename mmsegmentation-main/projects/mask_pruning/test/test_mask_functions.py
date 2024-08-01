import pytest
import sys
import torch

from ..utils.mask_functions import logistic_function, get_index_matrix




def test_logistic_function():
    res_one = logistic_function(torch.tensor([0]), 7, -1)
    assert abs(1-res_one) < 0.001

    # Check numerical stability
    res_one = logistic_function(torch.tensor([0]), 7, -100000)
    assert abs(1 - res_one) < 0.001

    res_zero = logistic_function(torch.tensor([0]), 7, 2)
    assert abs(res_zero) < 0.001

    # Check numerical stability
    res_zero = logistic_function(torch.tensor([0]), 7, 100000)
    assert abs(res_zero) < 0.001

def test_index_matrix():
    target = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    index_mat = get_index_matrix(3, 3, device="cpu")
    assert torch.all(torch.eq(target, index_mat))