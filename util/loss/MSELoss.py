from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class MSELoss(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_true - y_pred) ** 2)
