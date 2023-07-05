import json
import time
from shutil import copyfile

from tqdm import tqdm

from model.registration.mask_util import process_mask
from model.registration.model_util import get_reg_model
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.registration.SpatialNetwork import SpatialTransformer
import torch.nn.functional as F
from util.Optimizer import Optimizer
from util.ModelSaver import ModelSaver
from util.data_util.dataset import PairDataset, RandomPairDataset, ValPairDataset
from util.data_util.dataset1 import DynamicPairDataset
from util.visual.tensorboard_visual import visual_gradient, tensorboard_visual_registration, tensorboard_visual_dvf, \
    tensorboard_visual_det, tensorboard_visual_RGB_dvf, tensorboard_visual_deformation_2, tensorboard_visual_warp_grid
from util.util import update_dict
from util.util import mean_dict
from util.loss.GradientLoss import GradientLoss
from util.loss.BendingEnergyLoss import BendingEnergyLoss
from util.metric.registration_metric import folds_count_metric, folds_percent_metric, mse_metric, jacobian_determinant, \
    ssim_metric
from util.metric.segmentation_metric import dice_metric, ASD_metric, HD_metric, dice_metric2, ASD_metric2, HD_metric2
from util.visual.tensorboard_visual import tensorboard_visual_deformation
from util import loss
from util.visual.visual_registration import mk_grid_img


def train_reg_sup(config, basedir):
    print(f"start train time: {time.asctime()}")
    train_path = os.path.abspath(__file__)
    print(f"train file path: {train_path}")
    copyfile(os.path.abspath(__file__), os.path.join(basedir, os.path.basename(train_path)))
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
    # train_dataset = PairDataset(dataset_config, 'train_pair')
    # val_dataset = PairDataset(dataset_config, 'val_pair')

    seg = dataset_config[config['TrainConfig']['seg']] if len(config['TrainConfig']['seg']) > 0 else None
    if dataset_config['registration_type'] == 1 or dataset_config['registration_type'] == 2:
        atlas = dataset_config['atlas']
    else:
        atlas = None
    train_dataset = RandomPairDataset(dataset_config, dataset_type='train',
                                      registration_type=dataset_config['registration_type'], seg=seg, atlas=atlas)
    val_dataset = ValPairDataset(dataset_config, dataset_type='val',
                                 registration_type=dataset_config['registration_type'], atlas=atlas)

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
    reg_net = get_reg_model(config['ModelConfig'], checkpoint, dataset_config['image_size'])
    STN_bilinear = SpatialTransformer(dataset_config['image_size'], mode='bilinear')
    STN_nearest = SpatialTransformer(dataset_config['image_size'], mode='nearest')

    total = sum([param.nelement() for param in reg_net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    print(f'gpu: {config["TrainConfig"]["gpu"]}')
    device = torch.device("cuda:0" if len(config["TrainConfig"]["gpu"]) > 0 else "cpu")
    reg_net.to(device)
    STN_bilinear.to(device)
    STN_nearest.to(device)
    gpu_num = len(config["TrainConfig"]["gpu"].split(","))
    if gpu_num > 1:
        reg_net = torch.nn.DataParallel(reg_net, device_ids=[i for i in range(gpu_num)])
        STN_bilinear = torch.nn.DataParallel(STN_bilinear, device_ids=[i for i in range(gpu_num)])
        STN_nearest = torch.nn.DataParallel(STN_nearest, device_ids=[i for i in range(gpu_num)])

    optimizer = Optimizer(config=config["OptimConfig"],
                          model=reg_net,
                          checkpoint=config["TrainConfig"]["checkpoint"])
    model_without_ddp = reg_net.module if gpu_num > 1 else reg_net
    raw_grid_img = mk_grid_img(4, 1, dataset_config['image_size']).to(device)
    loss_function_dict = get_loss_function(config)
    step = 0
    best_val_dice_metric = -1. * float('inf')

    start_epoch = torch.load(checkpoint).get("epoch", 1) + 1 if checkpoint is not None else 1
    dynamic_dataset = DynamicPairDataset()
    supervised_dataset = dict()
    for epoch in range(start_epoch, config["TrainConfig"]["epoch"] + 1):
        """
         ------------------------------------------------
                 training network
         ------------------------------------------------
        """
        reg_net.train()
        train_losses_dict = dict()
        train_metrics_dict = dict()
        for id1, volume1, label1, id2, volume2, label2 in tqdm(train_dataloader):
            volume1 = volume1.to(device)
            label1 = label1.to(device) if label1 != [] else label1
            volume2 = volume2.to(device)
            label2 = label2.to(device) if label2 != [] else label2

            optimizer.zero_grad()

            dvf = reg_net(volume1, volume2)

            loss_dict = compute_reg_loss(config, dvf, loss_function_dict, STN_bilinear, volume1, label1, volume2,
                                         label2)

            loss_dict['total_loss'].backward()
            optimizer.step()
            update_dict(train_losses_dict, loss_dict)
            step += 1
            for i in range(volume1.shape[0]):
                supervised_dataset[id1[i]] = dict()
                supervised_dataset[id1[i]]['id1'] = id1[i]
                supervised_dataset[id1[i]]['volume1'] = volume1[i].detach().cpu()
                supervised_dataset[id1[i]]['label1'] = label1[i].detach().cpu() if label1 != [] else []
                supervised_dataset[id1[i]]['id2'] = id2[i]
                supervised_dataset[id1[i]]['volume2'] = STN_bilinear(volume1.clone().detach(),
                                                                     dvf.clone().detach()).cpu()[i]
                label2_ = STN_nearest(label1.clone().detach().float(), dvf.clone().detach()).cpu().type_as(
                    label1)[i] if label1 != [] else []
                supervised_dataset[id1[i]]['label2'] = label2_
                supervised_dataset[id1[i]]['dvf'] = dvf[i].clone().detach().cpu()

        mean_train_loss_dict = mean_dict(train_losses_dict)

        print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
        for loss_type, loss_value in mean_train_loss_dict.items():
            writer.add_scalar("train/loss/" + loss_type, loss_value, epoch)
        print(f"training total loss: {mean_train_loss_dict['total_loss']}")

        # mean_train_metric_dict = mean_dict(train_metrics_dict)
        # for metric_type, metric_value in mean_train_metric_dict.items():
        #     writer.add_scalar("train/metric/" + metric_type, metric_value, epoch)
        # print(f"training dice: {mean_train_metric_dict['dice']}")
        writer.add_scalar("lr", optimizer.get_cur_lr(), epoch)

        # visual_gradient(reg_net, writer, epoch)

        dynamic_dataset.update(supervised_dataset)
        dynamic_dataloader = DataLoader(dynamic_dataset, config["TrainConfig"]['batchsize'], shuffle=True)
        train_sup_losses_dict = dict()
        for id1, volume1, label1, id2, volume2, label2, target_dvf in tqdm(dynamic_dataloader):
            volume1 = volume1.to(device)
            label1 = label1.to(device) if label1 != [] else label1

            volume2 = volume2.to(device)
            label2 = label2.to(device) if label2 != [] else label2
            target_dvf = target_dvf.to(device)

            optimizer.zero_grad()

            dvf = reg_net(volume1, volume2)

            loss_dict = compute_reg_loss2(config, dvf, loss_function_dict, STN_bilinear, volume1, label1, volume2,
                                          label2, target_dvf)

            loss_dict['total_loss'].backward()
            optimizer.step()
            update_dict(train_sup_losses_dict, loss_dict)
            step += 1
        mean_train_sup_losses_dict = mean_dict(train_sup_losses_dict)
        for loss_type, loss_value in mean_train_sup_losses_dict.items():
            writer.add_scalar("train/sup_loss/" + loss_type, loss_value, epoch)
        print(f"training supervised total loss: {mean_train_sup_losses_dict['total_loss']}")

        """
         ------------------------------------------------
                validating network
         ------------------------------------------------
        """

        if epoch % config['TrainConfig']['val_interval'] == 0:
            reg_net.eval()
            val_losses_dict = dict()
            val_metrics_dict = dict()
            val_count = 0
            with torch.no_grad():
                for id1, volume1, label1, id2, volume2, label2 in tqdm(val_dataloader):
                    volume1 = volume1.to(device)
                    label1 = label1.to(device)
                    volume2 = volume2.to(device)
                    label2 = label2.to(device)
                    dvf = reg_net(volume1, volume2)

                    loss_dict = compute_reg_loss(config, dvf, loss_function_dict, STN_bilinear, volume1, label1,
                                                 volume2, label2)

                    update_dict(val_losses_dict, loss_dict)

                    warp_volume1 = STN_bilinear(volume1, dvf)
                    warp_label1 = STN_nearest(label1.float(), dvf).type(torch.uint8)

                    metric_dict = compute_reg_metric(dvf.clone().detach(), warp_volume1.clone().detach(),
                                                     warp_label1.clone().detach(),
                                                     volume2.clone().detach(), label2.clone().detach(), num_classes)

                    update_dict(val_metrics_dict, metric_dict)
            mean_val_loss_dict = mean_dict(val_losses_dict)
            mean_val_metric_dict = mean_dict(val_metrics_dict)

            print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
            for loss_type, loss_value in mean_val_loss_dict.items():
                writer.add_scalar("val/loss/" + loss_type, loss_value, epoch)
            print(f"val total loss: {mean_val_loss_dict['total_loss']}")
            for metric_type, metric_value in mean_val_metric_dict.items():
                writer.add_scalar("val/metric/" + metric_type, metric_value, epoch)
            print(f"val dice: {mean_val_metric_dict['dice']}")

            if epoch % 100 == 0:
                model_saver.save(os.path.join(basedir, "checkpoint", f"{epoch}_epoch.pth"),
                                 {"epoch": epoch, "model": model_without_ddp.state_dict(),
                                  "optim": optimizer.state_dict()})

            if mean_val_metric_dict['dice'] > best_val_dice_metric:
                best_val_dice_metric = mean_val_metric_dict['dice']
                model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                 {"epoch": epoch, "model": model_without_ddp.state_dict(),
                                  "optim": optimizer.state_dict()})
    model_saver.save(
        os.path.join(basedir, "checkpoint", "last_epoch.pth"),
        {"epoch": config["TrainConfig"]["epoch"], "model": model_without_ddp.state_dict(),
         "optim": optimizer.state_dict()})
    print(f"end train time: {time.asctime()}")


def compute_reg_loss(config, dvf, loss_function_dict, STN_bilinear, volume1, label1, volume2, label2):
    loss_dict = {}

    if config['LossConfig']['similarity_loss']['use']:
        loss_dict['similarity_loss'] = loss_function_dict['similarity_loss'](STN_bilinear(volume1, dvf), volume2)

    if config['LossConfig']['segmentation_loss']['use']:
        loss_dict['segmentation_loss'] = 0.
        if label1 != [] and label2 != []:
            num_classes = torch.max(label1).item()
            count = 0
            for i in range(1, num_classes + 1):
                label1_i = (label1 == i).float()
                label2_i = (label2 == i).float()
                if torch.any(label1_i) and torch.any(label2_i):
                    loss_dict['segmentation_loss'] = loss_dict['segmentation_loss'] + loss_function_dict[
                        'segmentation_loss'](STN_bilinear(label1_i, dvf), label2_i)
                    count = count + 1
            loss_dict['segmentation_loss'] = loss_dict['segmentation_loss']

    if config['LossConfig']['gradient_loss']['use']:
        loss_dict['gradient_loss'] = loss_function_dict['gradient_loss'](dvf)

    if config['LossConfig']['bending_energy_loss']['use']:
        loss_dict['bending_energy_loss'] = loss_function_dict['bending_energy_loss'](dvf)

    total_loss = 0.
    for loss_type in loss_dict.keys():
        total_loss = total_loss + config['LossConfig'][loss_type]['weight'] * loss_dict[loss_type]
    loss_dict['total_loss'] = total_loss

    return loss_dict


def compute_reg_loss2(config, dvf, loss_function_dict, STN_bilinear, volume1, label1, volume2, label2, target_dvf):
    loss_dict = {}

    if config['LossConfig']['similarity_loss']['use']:
        loss_dict['similarity_loss'] = loss_function_dict['similarity_loss'](STN_bilinear(volume1, dvf), volume2)

    if config['LossConfig']['segmentation_loss']['use']:
        loss_dict['segmentation_loss'] = 0.
        if label1 != [] and label2 != []:
            num_classes = torch.max(label1).item()
            count = 0
            for i in range(1, num_classes + 1):
                label1_i = (label1 == i).float()
                label2_i = (label2 == i).float()
                if torch.any(label1_i) and torch.any(label2_i):
                    loss_dict['segmentation_loss'] = loss_dict['segmentation_loss'] + loss_function_dict[
                        'segmentation_loss'](STN_bilinear(label1_i, dvf), label2_i)
                    count = count + 1
            loss_dict['segmentation_loss'] = loss_dict['segmentation_loss']

    if config['LossConfig']['gradient_loss']['use']:
        loss_dict['gradient_loss'] = loss_function_dict['gradient_loss'](dvf)

    if config['LossConfig']['bending_energy_loss']['use']:
        loss_dict['bending_energy_loss'] = loss_function_dict['bending_energy_loss'](dvf)

    loss_dict['supervised_loss'] = F.mse_loss(dvf, target_dvf, reduction='mean')

    total_loss = 0.
    for loss_type in loss_dict.keys():
        total_loss = total_loss + config['LossConfig'][loss_type]['weight'] * loss_dict[loss_type]

    loss_dict['total_loss'] = total_loss
    return loss_dict


def get_loss_function(config):
    loss_function_dict = dict()

    if config['LossConfig']['similarity_loss']['use']:
        loss_function_dict['similarity_loss'] = getattr(loss, config['LossConfig']['similarity_loss']['type'])()

    if config['LossConfig']['segmentation_loss']['use']:
        loss_function_dict['segmentation_loss'] = getattr(loss,
                                                          config['LossConfig']['segmentation_loss']['type'])()

    if config['LossConfig']['gradient_loss']['use']:
        loss_function_dict['gradient_loss'] = getattr(loss,
                                                      config['LossConfig']['gradient_loss']['type'])()

    if config['LossConfig']['bending_energy_loss']['use']:
        loss_function_dict['bending_energy_loss'] = getattr(loss,
                                                            config['LossConfig']['bending_energy_loss']['type'])()

    return loss_function_dict


def compute_reg_metric(dvf, warp_volume1, warp_label1, volume2, label2, num_classes):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['ASD'] = ASD_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['HD'] = HD_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['SSIM'] = ssim_metric(warp_volume1, volume2)
    metric_dict['folds_percent'] = folds_percent_metric(dvf).item()
    # metric_dict['folds_count'] = folds_count_metric(dvf).item()
    # metric_dict['mse'] = mse_metric(warp_volume1, volume2).item()
    return metric_dict
