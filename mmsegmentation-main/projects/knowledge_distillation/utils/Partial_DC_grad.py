# Code is copied from https://github.com/zhenxingjian/Partial_Distance_Correlation/tree/main/Partial_Distance_Correlation

import os
import sys
import time
import math
from builtins import isinstance

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from mmengine import MODELS


def Distance_Correlation(latent, control):

    latent = F.normalize(latent)
    control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim = -1) + 1e-12)

    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    
    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r

def P_Distance_Matrix(latent):
    n = latent.shape[0]
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1)  + 1e-18)
    aa = torch.sum(matrix_a, dim = 0, keepdims= True)/(n-2)
    ab = torch.sum(matrix_a, dim=0, keepdims=True) / (n - 2)
    ac = torch.sum(matrix_a)/((n-1)*(n-2))
    matrix_A = matrix_a - aa - ab + ac

    diag_A = torch.diag(torch.diag(matrix_A) ) 
    matrix_A = matrix_A - diag_A
    return matrix_A


def bracket_op(matrix_A, matrix_B):
    n = matrix_A.shape[0]
    return torch.sum(matrix_A * matrix_B)/(n*(n-3))


def P_removal(matrix_A, matrix_C):
    result = matrix_A - bracket_op(matrix_A, matrix_C) / bracket_op(matrix_C, matrix_C) * matrix_C
    return result

def Correlation(matrix_A, matrix_B):
    Gamma_XY = bracket_op(matrix_A, matrix_B)
    Gamma_XX = bracket_op(matrix_A, matrix_A)
    Gamma_YY = bracket_op(matrix_B, matrix_B)

    
    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-18)

    return correlation_r


def P_DC(latent_A, latent_B, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_B = P_Distance_Matrix(latent_B)
    matrix_GT = P_Distance_Matrix(ground_truth)

    # breakpoint()

    matrix_A_B = P_removal(matrix_A, matrix_B)
    # breakpoint()
    cr = Correlation(matrix_A_B, matrix_GT)

    return cr


def New_DC(latent_A, ground_truth):
    matrix_A = P_Distance_Matrix(latent_A)
    matrix_GT = P_Distance_Matrix(ground_truth)
    cr = Correlation(matrix_A, matrix_GT)

    return cr

class Loss_DC(nn.Module):
    def __init__(self):
        super(Loss_DC, self).__init__()

    def forward(self, featuresX, featuresY):

        if not isinstance(featuresX, list) and not isinstance(featuresX, tuple):
            featuresX = [featuresX]
        if not isinstance(featuresY, list) and not isinstance(featuresY, tuple):
            featuresY = [featuresY]

        dc_loss = 0
        for f_x, f_y in zip(featuresX, featuresY):
            f_x = f_x.reshape(f_x.shape[0], -1)
            f_y = f_y.reshape(f_y.shape[0], -1)

            matrix_X = P_Distance_Matrix(f_x)
            matrix_Y = P_Distance_Matrix(f_y)
            dc_loss = dc_loss - Correlation(matrix_X, matrix_Y)

        return dc_loss