import json
from tqdm import tqdm
from model.registration.model_util import get_reg_model
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.registration.SpatialNetwork import SpatialTransformer
import torch.nn.functional as F
from util.Optimizer import Optimizer
from util.ModelSaver import ModelSaver
from util.data_util.dataset import PairDataset
from util.seg_reg_util import create_train_dataset
from util.visual.tensorboard_visual import visual_gradient, tensorboard_visual_registration, tensorboard_visual_dvf, \
    tensorboard_visual_det, tensorboard_visual_RGB_dvf, tensorboard_visual_deformation_2, tensorboard_visual_warp_grid
from util.util import update_dict
from util.util import mean_dict
from util.loss.GradientLoss import GradientLoss
from util.loss.BendingEnergyLoss import BendingEnergyLoss
from util.metric.registration_metric import folds_count_metric, folds_percent_metric, mse_metric, jacobian_determinant, \
    ssim_metric
from util.metric.segmentation_metric import dice_metric, ASD_metric, HD_metric
from util.visual.tensorboard_visual import tensorboard_visual_deformation
from util import loss
from util.visual.visual_registration import mk_grid_img
import numpy as np


def train_reg_semi_supervised(config, basedir):
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

    reg_batch_generator_train, _ = create_train_dataset(dataset_config, config, seg=seg)

    val_dataset = PairDataset(dataset_config, 'val_pair')
    print(f'val dataset size: {len(val_dataset)}')
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
    raw_grid_img = mk_grid_img(4, 1, dataset_config['image_size']).to(device)
    loss_function_dict = get_loss_function(config)
    step = 0
    best_val_dice_metric = -1. * float('inf')
    num_batches_to_sample = len(dataset_config["train_pair"]) // config["TrainConfig"]['batchsize']

    for epoch in range(1, config["TrainConfig"]["epoch"] + 1):
        """
         ------------------------------------------------
                 training network
         ------------------------------------------------
        """
        reg_net.train()
        train_losses_dict = dict()
        train_dice_list = []
        with tqdm(total=num_batches_to_sample) as pbar:
            for id1, volume1, label1, id2, volume2, label2 in reg_batch_generator_train(num_batches_to_sample):
                volume1 = volume1.to(device)
                label1 = label1.to(device) if label1 != [] else label1

                volume2 = volume2.to(device)
                label2 = label2.to(device) if label2 != [] else label2

                optimizer.zero_grad()

                dvf = reg_net(volume1, volume2)

                loss_dict = compute_reg_loss2(config, dvf, loss_function_dict, STN_bilinear, volume1, label1, volume2,
                                              label2)

                if label1 != [] and label2 != []:
                    warp_label1 = STN_nearest(label1.float(), dvf)
                    axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
                    warp_label1_one_hot = F.one_hot(warp_label1.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        axis_order).contiguous()
                    label2_one_hot = F.one_hot(label2.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        axis_order).contiguous()
                    train_dice = dice_metric(warp_label1_one_hot, label2_one_hot).item()
                    train_dice_list.append(train_dice)
                    del warp_label1, warp_label1_one_hot, label2_one_hot

                loss_dict['total_loss'].backward()

                optimizer.step()

                update_dict(train_losses_dict, loss_dict)

                step += 1
                pbar.update(1)
        mean_train_loss_dict = mean_dict(train_losses_dict)
        print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
        for loss_type, loss_value in mean_train_loss_dict.items():
            writer.add_scalar("train/loss/" + loss_type, loss_value, epoch)
        print(f"training total loss: {mean_train_loss_dict['total_loss']}")
        writer.add_scalar("lr", optimizer.get_cur_lr(), epoch)
        if train_dice_list != []:
            mean_train_dice = np.mean(train_dice_list)
            print(f"training dice: {mean_train_dice}")
            writer.add_scalar("train/dice/", mean_train_dice, epoch)
        visual_gradient(reg_net, writer, epoch)
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

                    loss_dict = compute_reg_loss2(config, dvf, loss_function_dict, STN_bilinear, volume1, label1,
                                                  volume2, label2)

                    update_dict(val_losses_dict, loss_dict)

                    warp_volume1 = STN_bilinear(volume1, dvf)
                    warp_label1 = STN_nearest(label1.float(), dvf)

                    axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
                    warp_label1_one_hot = F.one_hot(warp_label1.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        axis_order).contiguous()
                    label2_one_hot = F.one_hot(label2.squeeze(dim=1).long(), num_classes=num_classes).permute(
                        axis_order).contiguous()

                    metric_dict = compute_reg_metric(dvf.clone().detach(), warp_volume1.clone().detach(),
                                                     warp_label1_one_hot.clone().detach(),
                                                     volume2.clone().detach(), label2_one_hot.clone().detach())
                    update_dict(val_metrics_dict, metric_dict)

                    if val_count < 8:
                        # grid_img = raw_grid_img.repeat(dvf.shape[0], 1, 1, 1, 1)
                        # warp_grid = STN_bilinear(grid_img.float(), dvf)
                        tensorboard_visual_registration(mode='val', name=id1[0] + '_' + id2[0] + '/img', writer=writer,
                                                        step=epoch, fix=volume2[0][0].detach().cpu(),
                                                        mov=volume1[0][0].detach().cpu(),
                                                        reg=warp_volume1[0][0].detach().cpu())
                        tensorboard_visual_registration(mode='val', name=id1[0] + '_' + id2[0] + '/seg', writer=writer,
                                                        step=epoch, fix=label2[0][0].detach().cpu(),
                                                        mov=label1[0][0].detach().cpu(),
                                                        reg=warp_label1[0][0].detach().cpu())
                        tag = 'val/' + id1[0] + '_' + id2[0]
                        tensorboard_visual_deformation(name=tag + '/deformation', dvf=dvf[0].detach().cpu(),
                                                       grid_spacing=dvf.shape[-1] // 50, writer=writer, step=epoch,
                                                       linewidth=1,
                                                       color='darkblue')
                        tensorboard_visual_dvf(name=tag + '/dvf', dvf=dvf[0].detach().cpu(), writer=writer, step=epoch)
                        tensorboard_visual_RGB_dvf(name=tag + '/rgb_dvf', dvf=dvf[0].detach().cpu(), writer=writer,
                                                   step=epoch)

                        det = jacobian_determinant(dvf[0].detach().cpu())

                        tensorboard_visual_det(name=tag + '/det', det=det, writer=writer, step=epoch,
                                               normalize_by='slice',
                                               threshold=0, cmap='Blues')
                        # tensorboard_visual_warp_grid(name=tag + '/warp_grid', warp_grid=warp_grid[0][0].cpu(),
                        #                              writer=writer,
                        #                              step=epoch)
                    val_count += 1
            mean_val_loss_dict = mean_dict(val_losses_dict)
            mean_val_metric_dict = mean_dict(val_metrics_dict)

            print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
            for loss_type, loss_value in mean_val_loss_dict.items():
                writer.add_scalar("val/loss/" + loss_type, loss_value, epoch)
            print(f"val total loss: {mean_val_loss_dict['total_loss']}")
            for metric_type, metric_value in mean_val_metric_dict.items():
                writer.add_scalar("val/metric/" + metric_type, metric_value, epoch)
            print(f"val dice: {mean_val_metric_dict['dice']}")

            if mean_val_metric_dict['dice'] > best_val_dice_metric:
                best_val_dice_metric = mean_val_metric_dict['dice']
                if gpu_num > 1:
                    model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                     {"model": reg_net.module.state_dict(), "optim": optimizer.state_dict()})
                else:
                    model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                     {"model": reg_net.state_dict(), "optim": optimizer.state_dict()})
    if gpu_num > 1:
        model_saver.save(
            os.path.join(basedir, "checkpoint", "last_epoch.pth"),
            {"model": reg_net.module.state_dict(), "optim": optimizer.state_dict()})
    else:
        model_saver.save(
            os.path.join(basedir, "checkpoint", "last_epoch.pth"),
            {"model": reg_net.state_dict(), "optim": optimizer.state_dict()})


def compute_reg_loss2(config, dvf, loss_function_dict, STN_bilinear, volume1, label1, volume2,
                      label2):
    loss_dict = {}

    if config['LossConfig']['similarity_loss']['use']:
        loss_dict['similarity_loss'] = loss_function_dict['similarity_loss'](STN_bilinear(volume1, dvf), volume2)

    if config['LossConfig']['segmentation_loss']['use']:
        loss_dict['segmentation_loss'] = 0.
        if label1 != [] and label2 != []:
            num_classes = torch.max(label1)
            for i in range(1, num_classes):
                loss_dict['segmentation_loss'] = loss_dict['segmentation_loss'] + loss_function_dict[
                    'segmentation_loss'](
                    STN_bilinear((label1 == i).float(), dvf), (label2 == i).float())

    if config['LossConfig']['gradient_loss']['use']:
        loss_dict['gradient_loss'] = loss_function_dict['gradient_loss'](dvf)

    if config['LossConfig']['bending_energy_loss']['use']:
        loss_dict['bending_energy_loss'] = loss_function_dict['bending_energy_loss'](dvf)

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


def compute_reg_metric(dvf, warp_volume1, warp_label1, volume2, label2):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(warp_label1, label2).item()
    metric_dict['ASD'] = ASD_metric(warp_label1, label2).item()
    metric_dict['HD'] = HD_metric(warp_label1, label2).item()
    metric_dict['SSIM'] = ssim_metric(warp_volume1, volume2)
    metric_dict['folds_percent'] = folds_percent_metric(dvf).item()
    return metric_dict
