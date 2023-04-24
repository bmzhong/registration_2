import json
import numpy as np
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
from util.data_util.dataset import PairDataset, SingleDataset

from util.visual.tensorboard_visual import tensorboard_visual_registration, tensorboard_visual_dvf, \
    tensorboard_visual_det, tensorboard_visual_segmentation, tensorboard_visual_RGB_dvf, \
    tensorboard_visual_deformation_2
from util.util import update_dict, std_dict
from util.util import mean_dict
from util.util import swap_training

from util.loss.GradientLoss import GradientLoss
from util.loss.BendingEnergyLoss import BendingEnergyLoss
from util.metric.registration_metric import folds_count_metric, folds_ratio_metric, mse_metric, jacobian_determinant
from util.metric.segmentation_metric import dice_metric
from util.visual.tensorboard_visual import tensorboard_visual_deformation
from util import loss
from util.seg_reg_util import create_train_dataset


def train_seg_reg(config, basedir):
    writer = SummaryWriter(log_dir=os.path.join(basedir, "logs"))
    reg_model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 2))
    seg_model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 2))

    """
     ------------------------------------------------
               preparing dataset
     ------------------------------------------------
    """
    with open(config['TrainConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)

    segmentation_available = dataset_config['segmentation_available']

    reg_batch_generator_train, seg_batch_generator_train = create_train_dataset(dataset_config, config,
                                                                                segmentation_available)

    reg_val_dataset = PairDataset(dataset_config, 'val_pair')

    seg_val_dataset = SingleDataset(dataset_config, 'val')

    reg_val_dataloader = DataLoader(reg_val_dataset, config['TrainConfig']['batchsize'], shuffle=False)
    seg_val_dataloader = DataLoader(seg_val_dataset, config['TrainConfig']['batchsize'], shuffle=False)

    """
     ------------------------------------------------
               loading model
     ------------------------------------------------
    """
    reg_checkpoint = None
    if len(config['TrainConfig']['Reg']['checkpoint']) > 0:
        reg_checkpoint = config['TrainConfig']['Reg']['checkpoint']
    reg_net = get_reg_model(config['ModelConfig']['Reg'], reg_checkpoint)
    STN_bilinear = SpatialTransformer(dataset_config['image_size'], mode='bilinear')
    STN_nearest = SpatialTransformer(dataset_config['image_size'], mode='nearest')

    seg_checkpoint = None
    if len(config['TrainConfig']['Seg']['checkpoint']) > 0:
        seg_checkpoint = config['TrainConfig']['Seg']['checkpoint']
    seg_net = get_seg_model(config['ModelConfig']['Seg'], dataset_config['region_number'] + 1, seg_checkpoint)

    reg_gpu = config['TrainConfig']['Reg']['gpu'].split(',')
    reg_device = torch.device("cuda:" + reg_gpu[0] if len(reg_gpu[0]) > 0 else 'cpu')
    reg_net.to(reg_device)
    STN_bilinear.to(reg_device)
    STN_nearest.to(reg_device)
    if len(reg_gpu) > 1:
        reg_net = torch.nn.DataParallel(reg_net, device_ids=[int(i) for i in reg_gpu])
        STN_bilinear = torch.nn.DataParallel(STN_bilinear, device_ids=[int(i) for i in reg_gpu])
        STN_nearest = torch.nn.DataParallel(STN_nearest, device_ids=[int(i) for i in reg_gpu])

    seg_gpu = config['TrainConfig']['Seg']['gpu'].split(',')
    seg_device = torch.device("cuda:" + seg_gpu[0] if len(seg_gpu[0]) > 0 else 'cpu')
    seg_net.to(seg_device)
    if len(seg_gpu) > 1:
        seg_net = torch.nn.DataParallel(seg_net, device_ids=[int(i) for i in seg_gpu])

    reg_optimizer = Optimizer(config=config["OptimConfig"]['Reg'],
                              model=reg_net,
                              checkpoint=config['TrainConfig']['Reg']['checkpoint'])
    seg_optimizer = Optimizer(config=config["OptimConfig"]['Seg'],
                              model=seg_net,
                              checkpoint=config['TrainConfig']['Seg']['checkpoint'])

    reg_step_per_epoch = config['TrainConfig']['Reg']['reg_step_per_epoch']
    seg_step_per_epoch = config['TrainConfig']['Seg']['seg_step_per_epoch']

    reg_loss_function_dict = get_reg_loss_function(config)

    seg_loss_function_dict = get_seg_loss_function(config)

    reg_best_val_metric = -1. * float('inf')
    seg_best_val_metric = -1. * float('inf')

    for epoch in range(1, config["TrainConfig"]["epoch"] + 1):

        """
         ------------------------------------------------
                training reg_net
         ------------------------------------------------
        """
        swap_training(network_to_train=reg_net, network_to_not_train=seg_net)
        reg_net.train()
        reg_train_losses_dict = dict()
        with tqdm(total=reg_step_per_epoch) as pbar:
            for id1, volume1, label1, id2, volume2, label2 in reg_batch_generator_train(reg_step_per_epoch):
                volume1 = volume1.to(reg_device)
                label1 = label1.to(reg_device) if label1 != [] else label1

                volume2 = volume2.to(reg_device)
                label2 = label2.to(reg_device) if label2 != [] else label2

                reg_optimizer.zero_grad()

                dvf = reg_net(volume1, volume2)

                loss_dict = compute_reg_loss1(config=config, dvf=dvf, reg_loss_function_dict=reg_loss_function_dict,
                                              seg_net=seg_net, STN_bilinear=STN_bilinear,
                                              volume1=volume1, label1=label1, volume2=volume2, label2=label2)

                loss_dict['total_loss'].backward()
                reg_optimizer.step()
                update_dict(reg_train_losses_dict, loss_dict)
                pbar.update(1)

        reg_mean_train_loss_dict = mean_dict(reg_train_losses_dict)
        print(f"\n\033[32mEpoch {epoch}/{config['TrainConfig']['epoch']}: \033[0m")
        for loss_type, loss_value in reg_mean_train_loss_dict.items():
            writer.add_scalar("train/reg/loss/" + loss_type, loss_value, epoch)
        print(f"\033[32mreg training total  loss: {reg_mean_train_loss_dict['total_loss']} \033[0m")
        writer.add_scalar("lr/reg", reg_optimizer.get_cur_lr(), epoch)

        """
         ------------------------------------------------
                validating reg_net
         ------------------------------------------------
        """

        if epoch % config['TrainConfig']['Reg']['val_interval'] == 0:
            reg_net.eval()
            reg_val_losses_dict = dict()
            reg_val_metrics_dict = dict()
            with torch.no_grad():
                for id1, volume1, label1, id2, volume2, label2 in reg_val_dataloader:
                    volume1 = volume1.to(reg_device)
                    label1 = label1.to(reg_device)
                    volume2 = volume2.to(reg_device)
                    label2 = label2.to(reg_device)
                    dvf = reg_net(volume1, volume2)

                    loss_dict = compute_reg_loss1(config=config, dvf=dvf, reg_loss_function_dict=reg_loss_function_dict,
                                                  seg_net=seg_net, STN_bilinear=STN_bilinear,
                                                  volume1=volume1, label1=label1, volume2=volume2, label2=label2)

                    update_dict(reg_val_losses_dict, loss_dict)

                    warp_volume1 = STN_bilinear(volume1, dvf)
                    warp_label1 = STN_nearest(label1.float(), dvf)

                    axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
                    warp_label1_one_hot = F.one_hot(warp_label1.squeeze(dim=1).long()).permute(axis_order).contiguous()
                    label2_one_hot = F.one_hot(label2.squeeze(dim=1).long()).permute(axis_order).contiguous()

                    metric_dict = compute_reg_metric(dvf.clone().detach(), warp_volume1.clone().detach(),
                                                     warp_label1_one_hot.clone().detach(),
                                                     volume2.clone().detach(), label2_one_hot.clone().detach())
                    update_dict(reg_val_metrics_dict, metric_dict)

                    tensorboard_visual_registration(mode='val/reg/', name=id1[0] + '_' + id2[0] + '/img', writer=writer,
                                                    step=epoch, fix=volume2[0][0].detach().cpu(),
                                                    mov=volume1[0][0].detach().cpu(),
                                                    reg=warp_volume1[0][0].detach().cpu())
                    tensorboard_visual_registration(mode='val/reg/', name=id1[0] + '_' + id2[0] + '/seg', writer=writer,
                                                    step=epoch, fix=label2[0][0].detach().cpu(),
                                                    mov=label1[0][0].detach().cpu(),
                                                    reg=warp_label1[0][0].detach().cpu())

                    tag = 'val/reg/' + id1[0] + '_' + id2[0]

                    tensorboard_visual_deformation(name=tag + '/deformation', dvf=dvf[0].detach().cpu(),
                                                   grid_spacing=dvf.shape[-1] // 50, writer=writer, step=epoch,
                                                   linewidth=1,
                                                   color='darkblue')

                    tensorboard_visual_dvf(name=tag + '/dvf', dvf=dvf[0].detach().cpu(), writer=writer, step=epoch)
                    tensorboard_visual_RGB_dvf(name=tag + '/rgb_dvf', dvf=dvf[0].detach().cpu(), writer=writer,
                                               step=epoch)

                    det = jacobian_determinant(dvf[0].detach().cpu())

                    tensorboard_visual_det(name=tag + '/det', det=det, writer=writer, step=epoch, normalize_by='slice',
                                           threshold=0, cmap='gray')

            reg_mean_val_loss_dict = mean_dict(reg_val_losses_dict)
            reg_mean_val_metric_dict = mean_dict(reg_val_metrics_dict)
            print(f"\n\033[32mEpoch {epoch}/{config['TrainConfig']['epoch']}:\033[0m")
            for loss_type, loss_value in reg_mean_val_loss_dict.items():
                writer.add_scalar("val/reg/loss/" + loss_type, loss_value, epoch)

            print(f"\033[32mval reg total loss: {reg_mean_val_loss_dict['total_loss']}\033[0m")
            for metric_type, metric_value in reg_mean_val_metric_dict.items():
                writer.add_scalar("val/reg/metric/" + metric_type, metric_value, epoch)
            print(f"\033[32mval reg dice: {reg_mean_val_metric_dict['dice']} \033[0m")
            if reg_mean_val_metric_dict['dice'] > reg_best_val_metric:
                reg_best_val_metric = reg_mean_val_metric_dict['dice']
                if len(reg_gpu) > 1:
                    reg_model_saver.save(os.path.join(basedir, "checkpoint", "reg_best_epoch.pth"),
                                         {"model": reg_net.module.state_dict(), "optim": reg_net.state_dict()})
                else:
                    reg_model_saver.save(os.path.join(basedir, "checkpoint", "reg_best_epoch.pth"),
                                         {"model": reg_net.state_dict(), "optim": reg_net.state_dict()})
        """
         ------------------------------------------------
                training seg_net
         ------------------------------------------------
        """
        swap_training(network_to_train=seg_net, network_to_not_train=reg_net)

        seg_net.train()
        seg_train_losses_dict = dict()
        with tqdm(total=seg_step_per_epoch) as pbar:
            for id1, volume1, label1, id2, volume2, label2 in seg_batch_generator_train(seg_step_per_epoch):
                volume1 = volume1.to(reg_device)
                label1 = label1.to(reg_device) if label1 != [] else label1

                volume2 = volume2.to(reg_device)
                label2 = label2.to(reg_device) if label2 != [] else label2

                seg_optimizer.zero_grad()

                dvf = reg_net(volume1, volume2)

                label1_predict = seg_net(volume1)
                label2_predict = seg_net(volume2)

                label1_predict = F.softmax(label1_predict, dim=1)
                label2_predict = F.softmax(label2_predict, dim=1)
                loss_dict = compute_seg_loss1(config=config, dvf=dvf, seg_loss_function_dict=seg_loss_function_dict,
                                              STN_nearest=STN_nearest, label1=label1, label1_predict=label1_predict,
                                              label2=label2, label2_predict=label2_predict)
                loss_dict['total_loss'].backward()

                seg_optimizer.step()

                update_dict(seg_train_losses_dict, loss_dict)
                pbar.update(1)
        seg_mean_train_loss_dict = mean_dict(seg_train_losses_dict)
        print(f"\n\033[34mEpoch {epoch}/{config['TrainConfig']['epoch']}:\033[0m")

        for loss_type, loss_value in seg_mean_train_loss_dict.items():
            writer.add_scalar("train/seg/loss/" + loss_type, loss_value, epoch)
        print(f"\033[34mtrain seg total loss: {seg_mean_train_loss_dict['total_loss']} \033[0m")

        writer.add_scalar("lr/seg", seg_optimizer.get_cur_lr(), epoch)

        """
         ------------------------------------------------
                validating seg_net
         ------------------------------------------------
        """
        if epoch % config['TrainConfig']['Seg']['val_interval'] == 0:
            seg_net.eval()
            seg_val_losses = []
            seg_val_metrics = []
            with torch.no_grad():
                for id, volume, label in seg_val_dataloader:
                    volume = volume.to(seg_device)
                    label = label.to(seg_device)

                    predict = seg_net(volume)

                    predict_softmax = F.softmax(predict, dim=1)

                    axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
                    label_one_hot = F.one_hot(label.squeeze(dim=1).long()).permute(axis_order).contiguous()

                    loss = seg_loss_function_dict['supervised_loss'](predict_softmax, label_one_hot.float())
                    seg_val_losses.append(loss.item())

                    predict_one_hot = F.one_hot(torch.argmax(predict, dim=1).long()).permute(axis_order).contiguous()

                    dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach(),
                                       background=True)

                    seg_val_metrics.append(dice.item())

                    predict_argmax = torch.argmax(predict_softmax, dim=1, keepdim=True)

                    tensorboard_visual_segmentation(mode='val/seg/', name=id[0], writer=writer, step=epoch,
                                                    volume=volume[0][0].detach().cpu(),
                                                    predict=predict_argmax[0][0].detach().cpu(),
                                                    target=label[0][0].detach().cpu())

            seg_mean_val_loss = np.mean(seg_val_losses)
            seg_mean_val_metric = np.mean(seg_val_metrics)
            print(f"\n\033[34mEpoch {epoch}/{config['TrainConfig']['epoch']}:\033[0m")
            print(f"\033[34mval seg loss: {seg_mean_val_loss}\033[0m")
            print(f"\033[34mval seg dice:  {seg_mean_val_metric}\033[0m")
            writer.add_scalar("val/seg/loss", seg_mean_val_loss, epoch)
            writer.add_scalar("val/seg/dice", seg_mean_val_metric, epoch)
            if seg_mean_val_metric > seg_best_val_metric:
                seg_best_val_metric = seg_mean_val_metric
                if len(seg_gpu) > 1:
                    seg_model_saver.save(os.path.join(basedir, "checkpoint", "seg_best_epoch.pth"),
                                         {"model": seg_net.module.state_dict(), "optim": seg_optimizer.state_dict()})
                else:
                    seg_model_saver.save(os.path.join(basedir, "checkpoint", "seg_best_epoch.pth"),
                                         {"model": seg_net.state_dict(), "optim": seg_optimizer.state_dict()})

    if len(reg_gpu) > 1:
        reg_model_saver.save(
            os.path.join(basedir, "checkpoint", "reg_last_epoch.pth"),
            {"model": reg_net.module.state_dict(), "optim": reg_optimizer.state_dict()})
    else:
        reg_model_saver.save(
            os.path.join(basedir, "checkpoint", "reg_last_epoch.pth"),
            {"model": reg_net.state_dict(), "optim": reg_optimizer.state_dict()})
    if len(seg_gpu) > 1:
        seg_model_saver.save(
            os.path.join(basedir, "checkpoint", "seg_last_epoch.pth"),
            {"model": seg_net.module.state_dict(), "optim": seg_optimizer.state_dict()})
    else:
        seg_model_saver.save(
            os.path.join(basedir, "checkpoint", "seg_last_epoch.pth"),
            {"model": seg_net.state_dict(), "optim": seg_optimizer.state_dict()})


def compute_seg_loss1(config, dvf, seg_loss_function_dict, STN_nearest, label1, label1_predict, label2, label2_predict):
    loss_dict = {}
    if label1 != [] and label2 != []:
        axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
        label1_one_hot = F.one_hot(label1.squeeze(dim=1).long()).permute(axis_order).contiguous()
        label2_one_hot = F.one_hot(label2.squeeze(dim=1).long()).permute(axis_order).contiguous()
        supervised_loss = seg_loss_function_dict['supervised_loss'](label1_predict, label1_one_hot) + \
                          seg_loss_function_dict['supervised_loss'](label2_predict, label2_one_hot)
    elif label1 != []:
        axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
        label1_one_hot = F.one_hot(label1.squeeze(dim=1).long()).permute(axis_order).contiguous()
        supervised_loss = seg_loss_function_dict['supervised_loss'](label1_predict, label1_one_hot)
        label2_one_hot = label2_predict
    elif label2 != []:
        axis_order = (0, label2.dim() - 1) + tuple(range(1, label2.dim() - 1))
        label2_one_hot = F.one_hot(label2.squeeze(dim=1).long()).permute(axis_order).contiguous()
        supervised_loss = seg_loss_function_dict['supervised_loss'](label2_predict, label2_one_hot)
        label1_one_hot = label1_predict
    else:
        raise Exception('label error')
    anatomy_loss = seg_loss_function_dict['anatomy_loss'](STN_nearest(label1_one_hot.float(), dvf), label2_one_hot)
    loss_dict['supervised_loss'] = supervised_loss
    loss_dict['anatomy_loss'] = anatomy_loss
    total_loss = 0.
    for loss_type in loss_dict.keys():
        total_loss = total_loss + config['LossConfig']['Seg'][loss_type]['weight'] * loss_dict[loss_type]
    loss_dict['total_loss'] = total_loss
    return loss_dict


def compute_reg_loss1(config, dvf, reg_loss_function_dict, seg_net, STN_bilinear, volume1, label1, volume2,
                      label2):
    loss_dict = {}

    if config['LossConfig']['Reg']['similarity_loss']['use']:
        loss_dict['similarity_loss'] = reg_loss_function_dict['similarity_loss'](STN_bilinear(volume1, dvf), volume2)

    if config['LossConfig']['Reg']['segmentation_loss']['use']:
        loss_dict['segmentation_loss'] = anatomy_loss(dvf, STN_bilinear, seg_net,
                                                      reg_loss_function_dict['segmentation_loss'],
                                                      volume1, label1, volume2, label2)

    if config['LossConfig']['Reg']['gradient_loss']['use']:
        loss_dict['gradient_loss'] = reg_loss_function_dict['gradient_loss'](dvf)

    if config['LossConfig']['Reg']['bending_energy_loss']['use']:
        loss_dict['bending_energy_loss'] = reg_loss_function_dict['bending_energy_loss'](dvf)

    total_loss = 0.
    for loss_type in loss_dict.keys():
        total_loss = total_loss + config['LossConfig']['Reg'][loss_type]['weight'] * loss_dict[loss_type]
    loss_dict['total_loss'] = total_loss

    return loss_dict


def anatomy_loss(dvf, STN_bilinear, seg_net, segmentation_loss_function, volume1, label1, volume2, label2):
    if label1 != []:
        axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
        label1_new = F.one_hot(label1.squeeze(dim=1).long()).permute(axis_order).contiguous()
    else:
        label1_new = seg_net(volume1)
        label1_new = F.softmax(label1_new, dim=1)

    if label2 != []:
        axis_order = (0, label2.dim() - 1) + tuple(range(1, label2.dim() - 1))
        label2_new = F.one_hot(label2.squeeze(dim=1).long()).permute(axis_order).contiguous()
    else:
        label2_new = seg_net(volume2)
        label2_new = F.softmax(label2_new, dim=1)

    return segmentation_loss_function(STN_bilinear(label1_new.float(), dvf), label2_new)


def get_reg_loss_function(config):
    loss_function_dict = dict()

    if config['LossConfig']['Reg']['similarity_loss']['use']:
        loss_function_dict['similarity_loss'] = getattr(loss, config['LossConfig']['Reg']['similarity_loss']['type'])()

    if config['LossConfig']['Reg']['segmentation_loss']['use']:
        loss_function_dict['segmentation_loss'] = getattr(loss,
                                                          config['LossConfig']['Reg']['segmentation_loss']['type'])(
            **config['LossConfig']['Reg']['segmentation_loss']['params'])

    if config['LossConfig']['Reg']['gradient_loss']['use']:
        loss_function_dict['gradient_loss'] = getattr(loss,
                                                      config['LossConfig']['Reg']['gradient_loss']['type'])()

    if config['LossConfig']['Reg']['bending_energy_loss']['use']:
        loss_function_dict['bending_energy_loss'] = getattr(loss,
                                                            config['LossConfig']['Reg']['bending_energy_loss'][
                                                                'type'])()
    return loss_function_dict


def get_seg_loss_function(config):
    loss_function_dict = dict()

    loss_function_dict['supervised_loss'] = getattr(loss, config['LossConfig']['Seg']['supervised_loss']['type'])()

    loss_function_dict['anatomy_loss'] = getattr(loss, config['LossConfig']['Seg']['anatomy_loss']['type'])()

    return loss_function_dict


def compute_reg_metric(dvf, warp_volume1, warp_label1, volume2, label2):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(warp_label1, label2).item()
    metric_dict['folds_ratio'] = folds_ratio_metric(dvf).item()
    metric_dict['folds_count'] = folds_count_metric(dvf).item()
    metric_dict['mse'] = mse_metric(warp_volume1, volume2).item()
    return metric_dict
