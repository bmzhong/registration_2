import json

import monai
import numpy as np
from tqdm import tqdm

from model.segmentation.model_util import get_seg_model
import torch
import torch.nn.functional as F
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
from util.ModelSaver import ModelSaver
from util.data_util.dataset import SingleDataset
from torch.utils.data import DataLoader
from util.Optimizer import Optimizer
from model.segmentation.unet.MonaiUnet import MonAI_Unet
from util.loss.DiceLoss import DiceLoss
from util.metric.segmentation_metric import dice_metric, ASD_metric, HD_metric
from model.segmentation.unet.UNet import UNet
from torchvision.transforms import transforms

from util.util import update_dict, mean_dict
from util.visual.tensorboard_visual import tensorboard_visual_segmentation, visual_gradient, tensorboard_visual_dvf


def train_seg(config, basedir):
    writer = SummaryWriter(log_dir=os.path.join(basedir, "logs"))
    model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 2))
    """
     ------------------------------------------------
               preparing dataset
     ------------------------------------------------
    """
    with open(config['TrainConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    num_classes = dataset_config['region_number']
    seg = dataset_config[config['TrainConfig']['seg']] if len(config['TrainConfig']['seg']) > 0 else dict()
    data_names = [key for key, value in seg.items() if value]
    train_dataset = SingleDataset(dataset_config, 'train', data_names=data_names)
    val_dataset = SingleDataset(dataset_config, 'val')

    print(f'train dataset size: {len(train_dataset)}')
    print(f'val dataset size: {len(val_dataset)}')

    train_dataloader = DataLoader(train_dataset, config["TrainConfig"]['batchsize'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, config["TrainConfig"]['batchsize'], shuffle=False)
    step_size = config['OptimConfig']['lr_scheduler']['params']['step_size']
    train_size, batch_size = dataset_config['train_size'], config["TrainConfig"]['batchsize']
    config['OptimConfig']['lr_scheduler']['params']['step_size'] = int(step_size * train_size / batch_size)
    """
     ------------------------------------------------
               loading model
     ------------------------------------------------
    """
    checkpoint = None
    if len(config['TrainConfig']['checkpoint']) > 0:
        checkpoint = config['TrainConfig']['checkpoint']

    seg_net = get_seg_model(config['ModelConfig'], dataset_config['region_number'] + 1, checkpoint)

    total = sum([param.nelement() for param in seg_net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    print(f'gpu: {config["TrainConfig"]["gpu"]}')
    device = torch.device("cuda:0" if len(config["TrainConfig"]["gpu"]) > 0 else "cpu")
    seg_net.to(device)

    gpu_num = len(config["TrainConfig"]["gpu"].split(","))
    if gpu_num > 1:
        seg_net = torch.nn.DataParallel(seg_net, device_ids=[i for i in range(gpu_num)])
    optimizer = Optimizer(config=config["OptimConfig"],
                          model=seg_net,
                          checkpoint=config["TrainConfig"]["checkpoint"])
    loss_function = DiceLoss(background=True)
    step = 0
    best_val_metric = -1. * float('inf')
    start_epoch = torch.load(checkpoint).get("epoch", 1) if checkpoint is not None else 1
    for epoch in range(start_epoch, config["TrainConfig"]["epoch"] + 1):
        """
         ------------------------------------------------
                 training network
         ------------------------------------------------
        """

        seg_net.train()
        train_losses = []
        train_metrics_dict = dict()
        for id, volume, label in tqdm(train_dataloader):
            volume = volume.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            predict = seg_net(volume)

            predict_softmax = F.softmax(predict, dim=1)

            axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
            label_one_hot = F.one_hot(label.squeeze(dim=1).long(), num_classes=num_classes + 1).permute(
                axis_order).contiguous()

            loss = loss_function(predict_softmax, label_one_hot.float())

            loss.backward()

            optimizer.step()
            train_losses.append(loss.item())

            predict_one_hot = F.one_hot(torch.argmax(predict_softmax, dim=1).long(),
                                        num_classes=num_classes + 1).permute(axis_order).contiguous()

            metric_dict = compute_seg_metric1(predict_one_hot.clone().detach(), label_one_hot.clone().detach())

            update_dict(train_metrics_dict, metric_dict)

            step = step + 1

        mean_train_loss = np.mean(train_losses)
        mean_train_metric_dict = mean_dict(train_metrics_dict)
        print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
        print(f'training loss: {mean_train_loss}')
        print(f"training dice:  {mean_train_metric_dict['dice']}")
        writer.add_scalar("train/loss", mean_train_loss, epoch)
        for metric_type, metric_value in mean_train_metric_dict.items():
            writer.add_scalar("train/metric/" + metric_type, metric_value, epoch)
        writer.add_scalar("lr", optimizer.get_cur_lr(), epoch)
        visual_gradient(seg_net, writer, epoch)
        """
         ------------------------------------------------
                validating network
         ------------------------------------------------
        """
        if epoch % config['TrainConfig']['val_interval'] == 0:
            seg_net.eval()

            val_losses = []
            val_metrics_dict = dict()
            with torch.no_grad():
                for id, volume, label in tqdm(val_dataloader):
                    volume = volume.to(device)
                    label = label.to(device)

                    predict = seg_net(volume)

                    predict_softmax = F.softmax(predict, dim=1)

                    axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
                    label_one_hot = F.one_hot(label.squeeze(dim=1).long(), num_classes=num_classes + 1).permute(
                        axis_order).contiguous()

                    loss = loss_function(predict_softmax, label_one_hot.float())
                    val_losses.append(loss.item())

                    predict_one_hot = F.one_hot(torch.argmax(predict, dim=1).long(),
                                                num_classes=num_classes + 1).permute(
                        axis_order).contiguous()

                    metric_dict = compute_seg_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach())

                    update_dict(val_metrics_dict, metric_dict)

                    predict_argmax = torch.argmax(predict_softmax, dim=1, keepdim=True)
                    tensorboard_visual_segmentation(mode='val', name=id[0], writer=writer, step=epoch,
                                                    volume=volume[0][0].detach().cpu(),
                                                    predict=predict_argmax[0][0].detach().cpu(),
                                                    target=label[0][0].detach().cpu())
            print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
            mean_val_loss = np.mean(val_losses)
            mean_val_metric_dict = mean_dict(val_metrics_dict)
            print(f'val loss: {mean_val_loss}')
            print(f"val dice:  {mean_val_metric_dict['dice']}")
            writer.add_scalar("val/loss", mean_val_loss, epoch)
            for metric_type, metric_value in mean_val_metric_dict.items():
                writer.add_scalar("val/metric/" + metric_type, metric_value, epoch)

            if mean_val_metric_dict['dice'] > best_val_metric:
                best_val_metric = mean_val_metric_dict['dice']
                if gpu_num > 1:
                    model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                     {"epoch": epoch, "model": seg_net.module.state_dict(),
                                      "optim": optimizer.state_dict()})
                else:
                    model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                     {"epoch": epoch, "model": seg_net.state_dict(),
                                      "optim": optimizer.state_dict()})
    if gpu_num > 1:
        model_saver.save(
            os.path.join(basedir, "checkpoint", 'last_epoch.pth'),
            {"epoch": config["TrainConfig"]["epoch"], "model": seg_net.module.state_dict(),
             "optim": optimizer.state_dict()})
    else:
        model_saver.save(os.path.join(basedir, "checkpoint", "last_epoch.pth"),
                         {"epoch": config["TrainConfig"]["epoch"], "model": seg_net.state_dict(),
                          "optim": optimizer.state_dict()})

    del seg_net


def compute_seg_metric1(predict, target):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(predict, target).item()
    return metric_dict


def compute_seg_metric(predict, target):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(predict, target).item()
    metric_dict['ASD'] = ASD_metric(predict, target).item()
    metric_dict['HD'] = HD_metric(predict, target).item()
    return metric_dict
