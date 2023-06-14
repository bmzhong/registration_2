import json
from tqdm import tqdm
from model.registration.model_util import get_reg_model
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.registration.SpatialNetwork import SpatialTransformer
import torch.nn.functional as F

from model.segmentation.model_util import get_seg_model
from util.Optimizer import Optimizer
from util.ModelSaver import ModelSaver
from util.data_util.dataset import PairDataset, RandomPairDataset
from util.loss import DiceLoss
from util.visual.tensorboard_visual import visual_gradient, tensorboard_visual_registration, tensorboard_visual_dvf, \
    tensorboard_visual_det, tensorboard_visual_RGB_dvf, tensorboard_visual_deformation_2, tensorboard_visual_warp_grid, \
    tensorboard_visual_segmentation
from util.util import update_dict
from util.util import mean_dict
from util.metric.registration_metric import folds_count_metric, folds_percent_metric, mse_metric, jacobian_determinant, \
    ssim_metric
from util.metric.segmentation_metric import dice_metric, ASD_metric, HD_metric
from util.visual.tensorboard_visual import tensorboard_visual_deformation
from util import loss
from util.visual.visual_registration import mk_grid_img
import numpy as np


def train_seg_x(config, basedir):
    writer = SummaryWriter(log_dir=os.path.join(basedir, "logs"))
    model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 2))

    """
     ------------------------------------------------
               preparing dataset
     ------------------------------------------------
    """
    with open(config['TrainConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    num_classes = dataset_config['region_number'] + 1

    seg = dataset_config[config['TrainConfig']['seg']] if len(config['TrainConfig']['seg']) > 0 else None

    data_names = get_seg_data_names(dataset_config, seg)

    train_dataset = PairDataset(dataset_config, 'train_pair', data_names=data_names)
    val_dataset = PairDataset(dataset_config, 'val_pair')

    print(f'train dataset size: {len(train_dataset)}')
    print(f'val dataset size: {len(val_dataset)}')

    train_dataloader = DataLoader(train_dataset, config["TrainConfig"]['batchsize'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, config["TrainConfig"]['batchsize'], shuffle=False)

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
        for id1, volume1, label1, id2, volume2, label2 in tqdm(train_dataloader):
            volume1 = volume1.to(device)
            label1 = label1.to(device) if label1 != [] else label1

            volume2 = volume2.to(device)
            label2 = label2.to(device) if label2 != [] else label2

            optimizer.zero_grad()

            predict1, predict2 = seg_net(volume1, volume2)
            predict1_softmax = F.softmax(predict1, dim=1)
            predict2_softmax = F.softmax(predict2, dim=1)

            axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
            label1_one_hot = F.one_hot(label1.squeeze(dim=1).long(), num_classes=num_classes).permute(
                axis_order).contiguous()
            label2_one_hot = F.one_hot(label2.squeeze(dim=1).long(), num_classes=num_classes).permute(
                axis_order).contiguous()

            loss1 = loss_function(predict1_softmax, label1_one_hot.float())
            loss2 = loss_function(predict2_softmax, label2_one_hot.float())
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            predict1_one_hot = F.one_hot(torch.argmax(predict1_softmax, dim=1).long(),
                                         num_classes=num_classes).permute(axis_order).contiguous()
            predict2_one_hot = F.one_hot(torch.argmax(predict2_softmax, dim=1).long(),
                                         num_classes=num_classes).permute(axis_order).contiguous()

            metric_dict1 = compute_seg_metric1(predict1_one_hot.clone().detach(), label1_one_hot.clone().detach())
            metric_dict2 = compute_seg_metric1(predict2_one_hot.clone().detach(), label2_one_hot.clone().detach())

            update_dict(train_metrics_dict, metric_dict1)
            update_dict(train_metrics_dict, metric_dict2)

            step += 1
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
                for id1, volume1, label1, id2, volume2, label2 in tqdm(val_dataloader):
                    volume1 = volume1.to(device)
                    label1 = label1.to(device)
                    volume2 = volume2.to(device)
                    label2 = label2.to(device)
                    predict1, predict2 = seg_net(volume1, volume2)
                    predict1_softmax = F.softmax(predict1, dim=1)
                    predict2_softmax = F.softmax(predict2, dim=1)

                    axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
                    label1_one_hot = F.one_hot(label1.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        axis_order).contiguous()
                    label2_one_hot = F.one_hot(label2.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        axis_order).contiguous()

                    loss1 = loss_function(predict1_softmax, label1_one_hot.float())
                    loss2 = loss_function(predict2_softmax, label2_one_hot.float())
                    loss = loss1 + loss2
                    val_losses.append(loss.item())

                    predict1_one_hot = F.one_hot(torch.argmax(predict1_softmax, dim=1).long(),
                                                 num_classes=num_classes).permute(axis_order).contiguous()
                    predict2_one_hot = F.one_hot(torch.argmax(predict2_softmax, dim=1).long(),
                                                 num_classes=num_classes).permute(axis_order).contiguous()

                    metric_dict1 = compute_seg_metric1(predict1_one_hot.clone().detach(),
                                                       label1_one_hot.clone().detach())
                    metric_dict2 = compute_seg_metric1(predict2_one_hot.clone().detach(),
                                                       label2_one_hot.clone().detach())

                    update_dict(val_metrics_dict, metric_dict1)
                    update_dict(val_metrics_dict, metric_dict2)

                    predict1_argmax = torch.argmax(predict1_softmax, dim=1, keepdim=True)
                    predict2_argmax = torch.argmax(predict2_softmax, dim=1, keepdim=True)

                    tensorboard_visual_segmentation(mode='val', name=id1[0], writer=writer, step=epoch,
                                                    volume=volume1[0][0].detach().cpu(),
                                                    predict=predict1_argmax[0][0].detach().cpu(),
                                                    target=label1[0][0].detach().cpu())
                    tensorboard_visual_segmentation(mode='val', name=id1[0], writer=writer, step=epoch,
                                                    volume=volume2[0][0].detach().cpu(),
                                                    predict=predict2_argmax[0][0].detach().cpu(),
                                                    target=label2[0][0].detach().cpu())
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


def get_seg_data_names(dataset_config, seg):
    subdivide_names_dict = {'00': [], '01': [], '10': [], '11': []}
    train_data_pair_names = np.array(dataset_config['train_pair'])
    for name1, name2 in train_data_pair_names:
        if seg[name1] and seg[name2]:
            subdivide_names_dict['11'].append([name1, name2])
        elif not seg[name1] and seg[name2]:
            subdivide_names_dict['01'].append([name1, name2])
        elif seg[name1] and not seg[name2]:
            subdivide_names_dict['10'].append([name1, name2])
        else:
            subdivide_names_dict['00'].append([name1, name2])
    return subdivide_names_dict['11']


def compute_seg_metric1(predict, target):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(predict, target).item()
    return metric_dict
