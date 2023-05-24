import torch
import numpy as np
import random


# def random_mask(volume1, volume2, label1, label2):
#     D, H, W = volume1.shape[2:]
#     min_ratio = 1 / 4
#     max_ratio = 1 / 3
#     # ratio = np.random.uniform(min_ratio, max_ratio)
#     rD, rH, rW = np.random.randint(low=(0, 0, 0), high=(D, H, W))

def generate_segmentation_mask(volume1, volume2, label1, label2, num_classes):
    sample_number = np.random.randint(1, num_classes + 1)
    seg_ids_list = [i for i in range(1, num_classes + 1)]
    sample_seg_ids = np.random.choice(seg_ids_list, size=sample_number, replace=False)
    mask1 = torch.zeros(volume1.shape, device=volume1.device)
    mask2 = torch.zeros(volume2.shape, device=volume1.device)
    for id in sample_seg_ids:
        mask1 = mask1 + (label1 == id)
        mask2 = mask2 + (label2 == id)
    return mask1, mask2


def segmentation_mask(volume1, volume2, label1, label2, num_classes):
    if label1 == [] and label2 == []:
        return volume1, volume2, label1, label2
    mask1, mask2 = generate_segmentation_mask(volume1, volume2, label1, label2, num_classes)
    mask1 = 1 - mask1
    mask2 = 1 - mask2
    volume1 = volume1 * mask1
    label1 = label1 * mask1
    volume2 = volume2 * mask2
    label2 = label2 * mask2
    return volume1, volume2, label1.type(torch.uint8), label2.type(torch.uint8)


def neg_segmentation_mask(volume1, volume2, label1, label2, num_classes):
    if label1 == [] and label2 == []:
        return volume1, volume2, label1, label2
    mask1, mask2 = generate_segmentation_mask(volume1, volume2, label1, label2, num_classes)
    volume1 = volume1 * mask1
    label1 = label1 * mask1
    volume2 = volume2 * mask2
    label2 = label2 * mask2
    return volume1, volume2, label1.type(torch.uint8), label2.type(torch.uint8)


def process_mask(volume1, volume2, label1, label2, num_classes, mask_type, probability):
    if np.random.uniform() > probability:
        return volume1, volume2, label1, label2
    if mask_type == 'seg_mask':
        return segmentation_mask(volume1, volume2, label1, label2, num_classes)
    elif mask_type == 'neg_seg_mask':
        return neg_segmentation_mask(volume1, volume2, label1, label2, num_classes)
    return volume1, volume2, label1, label2
