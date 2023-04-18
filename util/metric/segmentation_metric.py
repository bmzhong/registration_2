import torch
import torch.nn.functional as F


def dice_coefficient(predict, target):
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

    return (2. * intersection) / (predict.sum() + target.sum())


def dice_metric(predict, target, background=True):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """

    assert predict.shape == target.shape, "predict and target must have the same shape"

    num_classes = target.shape[1]

    total_dice = 0.

    start = 0 if background else 1

    for i in range(start, num_classes):
        total_dice = total_dice + dice_coefficient(predict[:, i], target[:, i])

    mean_dice = total_dice / num_classes if background else total_dice / (num_classes - 1)

    return mean_dice
