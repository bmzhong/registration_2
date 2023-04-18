import os

import torch
from random import randint
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk

from util.visual.visual_registration import preview_3D_deformation, preview_3D_vector_field, preview_image


def tensorboard_visual_segmentation(mode, name, writer, step, volume, predict, target, interval=10):
    """
    mode: train/val/test
    name: image name
    writer: SummaryWriter
    predict:  1, D, H, W
    target:   1, D, H, W
    """

    # D, H, W
    volume = volume[0].cpu()
    predict = predict[0].cpu()
    target = target[0].cpu()

    title_name = name + '_volume, label, predict'

    img_list = [volume, target, predict]
    tag = mode + '/' + name
    visual_img_list(tag=tag, title_name=title_name, writer=writer, step=step, img_list=img_list, interval=interval)


def tensorboard_visual_registration(mode, name, writer, step, fix, mov, reg, interval=10):
    """
    mode: train/val/test
    name: image name
    writer: SummaryWriter
    predict:  1, D, H, W
    target:   1, D, H, W
    """

    # D, H, W
    fix = fix[0].cpu()
    mov = mov[0].cpu()
    reg = reg[0].cpu()
    title_name = name + '_fix, mov, reg'
    img_list = [fix, mov, reg]
    tag = mode + '/' + name
    visual_img_list(tag=tag, title_name=title_name, writer=writer, step=step, img_list=img_list, interval=interval)


def visual_img_list(tag, title_name, writer, step, img_list, interval):
    # 1, H*rows_number, W*2
    shape = img_list[0].shape

    img_slice = torch.cat(
        [torch.cat([img[i, :, :].unsqueeze(dim=0) for img in img_list], dim=2) for i in
         range(0, shape[0], interval)],
        dim=1)
    title_patch = create_header(img_slice.shape, title_name)
    writer.add_image(tag + '/D', torch.cat((title_patch, img_slice), dim=1), step)

    # 1, D*rows_number, W*2
    img_slice = torch.cat(
        [torch.cat([img[:, i, :].unsqueeze(dim=0) for img in img_list], dim=2) for i in
         range(0, shape[1], interval)],
        dim=1)
    title_patch = create_header(img_slice.shape, title_name)
    writer.add_image(tag + '/H', torch.cat((title_patch, img_slice), dim=1), step)

    # 1, D*rows_number, H*2
    img_slice = torch.cat(
        [torch.cat([img[:, :, i].unsqueeze(dim=0) for img in img_list], dim=2) for i in
         range(0, shape[2], interval)],
        dim=1)
    title_patch = create_header(img_slice.shape, title_name)
    writer.add_image(tag + '/W', torch.cat((title_patch, img_slice), dim=1), step)


def create_header(shape, name):
    header = np.zeros((100, shape[2], 1), dtype=np.uint8) + 255
    header = cv2.putText(header, name, (10, header.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, 0, 1)
    header = header.astype(np.float32) / 255
    header = np.transpose(header, (2, 0, 1))
    header = torch.Tensor(header)
    return header


def visual_gradient(model: Module, writer: SummaryWriter, step: int):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram("grad/" + name, param.grad, step)


def tensorboard_visual_deformation(name, dvf, grid_spacing, writer, step, **kwargs):
    figure = preview_3D_deformation(dvf, grid_spacing, **kwargs)
    writer.add_figure(name, figure, step)


def tensorboard_visual_dvf(name, dvf, writer, step):
    figure = preview_3D_vector_field(dvf)
    writer.add_figure(name, figure, step)


def tensorboard_visual_det(name, det, writer, step, **kwargs):
    figure = preview_image(det, **kwargs)
    writer.add_figure(name, figure, step)