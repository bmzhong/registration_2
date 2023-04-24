import monai
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, background=False, weight=None, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.background = background
        self.weight = weight
        self.smooth = smooth

    def forward(self, predict, target):
        """
        predict: B, C, D, H, W
        target   B, C, D, H, W
        """

        assert predict.shape == target.shape, "predict and target must have the same shape"

        num_classes = target.shape[1]

        total_dice = 0.

        weight = torch.tensor([1.] * num_classes) if self.weight is None else self.weight

        start = 0 if self.background else 1

        for i in range(start, num_classes):
            _dice = self.dice_coefficient(predict[:, i], target[:, i])
            total_dice = total_dice + weight[i] * _dice

        mean_dice = total_dice / num_classes if self.background else total_dice / (num_classes - 1)

        return 1. - mean_dice

    def dice_coefficient(self, predict, target):
        """
        predict: B, D, H, W
        target: B, D, H, W
        """
        B = predict.shape[0]

        # B, D, H, W -> B, D*H*W
        predict = predict.view(B, -1)

        # B, D, H, W -> B, D*H*W
        target = target.view(B, -1)

        intersection = (predict * target).sum()

        return (2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)

# if __name__ == '__main__':
# dice_loss = BinaryDiceLoss()
# predict = torch.tensor([
#     [0.5322, 0.4932, 0.1764],
#     [0.3107, 0.5297, 0.1604],
#     [0.3841, 0.3537, 0.3574],
#     [0.3323, 0.8301, 0.6436]
# ]).unsqueeze(0).unsqueeze(0)
#
# target = torch.tensor([[0] * 3, [0] * 3, [1] * 3, [1] * 3]).unsqueeze(0).unsqueeze(0)
# loss = dice_loss(predict, target)
# print(loss)
#
# dice_loss = MultiClassDiceLoss()
# predict = predict.repeat(10, 6, 1, 1)
# target = target.repeat(10, 1, 1, 1)
# loss = dice_loss(predict, target)
# print(loss)
#
# dice_loss2 = monai.losses.DiceLoss(
#     include_background=True,
#     to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
#     softmax=True,  # Note that our segmentation network is missing the softmax at the end
#     reduction="mean"
# )
# loss = dice_loss2(predict, target)
# print(loss)
