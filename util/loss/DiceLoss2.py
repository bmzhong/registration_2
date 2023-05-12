from torch import nn
import torch


class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()

    def forward(self, predict, target):
        B = predict.shape[0]
        numerator = torch.sum(predict * target, dim=(1, 2, 3, 4)) * 2
        denominator = torch.sum(predict, dim=(1, 2, 3, 4)) + torch.sum(target, dim=(1, 2, 3, 4)) + 1e-6
        return torch.sum(1 - numerator / denominator) / B