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


def dice_metric(predict, target):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """

    assert predict.shape == target.shape, "predict and target must have the same shape"

    num_classes = target.shape[1]
    total_dice = 0.
    count = 0

    for i in range(1, num_classes):
        if torch.any(target[:, i]):
            count = count + 1
            total_dice = total_dice + dice_coefficient(predict[:, i], target[:, i])

    return total_dice / count if count > 0 else 0.


def dice_metric2(predict, target, num_classes):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    total_dice = 0.
    count = 0
    for i in range(1, num_classes + 1):
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(target_i):
            count = count + 1
            total_dice = total_dice + dice_coefficient(predict_i, target_i)
    return total_dice / count if count > 0 else 0.


def ASD_metric(predict, target):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """
    num_classes = target.shape[1]
    total_ASD = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = predict[:, i].unsqueeze(dim=1)
        target_i = target[:, i].unsqueeze(dim=1)
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            ASD_i = monai.metrics.compute_average_surface_distance(predict_i, target_i, include_background=True)
            total_ASD = total_ASD + torch.mean(ASD_i)

    return total_ASD / count if count > 0 else torch.inf


def ASD_metric2(predict, target, num_classes):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    total_ASD = 0.
    count = 0
    for i in range(1, num_classes + 1):
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            ASD_i = monai.metrics.compute_average_surface_distance(predict_i, target_i, include_background=True)
            total_ASD = total_ASD + torch.mean(ASD_i)
    return total_ASD / count if count > 0 else torch.inf


def HD_metric(predict, target):
    """
    predict: B, C, D, H, W
    target:  B, C, D, H, W
    """
    num_classes = target.shape[1]
    total_HD = 0.
    count = 0
    for i in range(1, num_classes):
        predict_i = predict[:, i].unsqueeze(dim=1)
        target_i = target[:, i].unsqueeze(dim=1)
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            HD_i = monai.metrics.compute_hausdorff_distance(predict_i, target_i, include_background=True)
            total_HD = total_HD + torch.mean(HD_i)
    return total_HD / count if count > 0 else torch.inf


def HD_metric2(predict, target, num_classes):
    """
    predict: B, 1, D, H, W
    target:  B, 1, D, H, W
    """
    total_HD = 0.
    count = 0
    for i in range(1, num_classes + 1):
        predict_i = (predict == i).float()
        target_i = (target == i).float()
        if torch.any(predict_i) and torch.any(target_i):
            count = count + 1
            HD_i = monai.metrics.compute_hausdorff_distance(predict_i, target_i, include_background=True)
            total_HD = total_HD + torch.mean(HD_i)
    return total_HD / count if count > 0 else torch.inf


if __name__ == '__main__':
    a = torch.randint(0, 2, size=(2, 1, 128, 128, 128))
    b = torch.randint(0, 2, size=(2, 1, 128, 128, 128))
    print(ASD_metric(a, a))
