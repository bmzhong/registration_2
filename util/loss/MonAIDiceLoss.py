from typing import Callable, Union, Optional

import monai
import torch.nn as nn
from monai.utils import LossReduction


class MonAIDiceLoss(nn.Module):
    def __init__(self,
                 include_background: bool = True,
                 to_onehot_y: bool = False,
                 sigmoid: bool = False,
                 softmax: bool = False,
                 other_act: Optional[Callable] = None,
                 squared_pred: bool = False,
                 jaccard: bool = False,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN,
                 smooth_nr: float = 1e-5,
                 smooth_dr: float = 1e-5,
                 batch: bool = False,
                 ):
        super(MonAIDiceLoss, self).__init__()
        self.dice_loss = monai.losses.DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch
        )

    def forward(self, input, target):
        return self.dice_loss(input, target)
