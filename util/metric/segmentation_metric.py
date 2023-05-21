import torch
import torch.nn.functional as F
import monai


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

    two_sum = predict.sum() + target.sum()

    return 2. * intersection / two_sum if two_sum > 1e-6 else 0.

    # return (2. * intersection) / (predict.sum() + target.sum())


def dice_metric(predict, target, background=False):
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

    mean_dice = total_dice / \
        num_classes if background else total_dice / (num_classes - 1)

    return mean_dice


def ASD_metric(predict, target):
    ASD = monai.metrics.compute_average_surface_distance(
        predict, target, include_background=False, symmetric=False)
    return torch.mean(ASD)


def HD_metric(predict, target):
    HD = monai.metrics.compute_hausdorff_distance(
        predict, target, include_background=False)
    return torch.mean(HD)


if __name__ == '__main__':
    a = torch.randint(0, 2, size=(2, 1, 128, 128, 128))
    b = torch.randint(0, 2, size=(2, 1, 128, 128, 128))
    print(ASD_metric(a, a))
