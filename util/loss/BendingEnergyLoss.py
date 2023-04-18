import monai
from torch import nn
import torch


class BendingEnergyLoss(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super(BendingEnergyLoss, self).__init__()
        self.bending_loss = monai.losses.BendingEnergyLoss()

    def forward(self, displacement_vector_field):
        return self.bending_loss(displacement_vector_field)
