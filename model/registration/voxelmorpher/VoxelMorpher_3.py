import monai
import torch
from torch import nn


class VoxelMorpher_3(nn.Module):
    def __init__(self):
        super(VoxelMorpher_3, self).__init__()
        self.u_net = monai.networks.nets.UNet(
            3,  # spatial dims
            2,  # input channels (one for fixed image and one for moving image)
            3,  # output channels (to represent 3D displacement vector field)
            (16, 32, 32, 32, 32),  # channel sequence
            (1, 2, 2, 2),  # convolutional strides
            dropout=0.2,
            norm="batch"
        )

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.u_net(x)
        return x
