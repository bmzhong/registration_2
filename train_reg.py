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
from util.visual.tensorboard_visual import visual_gradient, tensorboard_visual_registration, tensorboard_visual_dvf, \
    tensorboard_visual_det
from util.util import update_dict
from util.util import mean_dict
from util.loss.GradientLoss import GradientLoss
from util.loss.BendingEnergyLoss import BendingEnergyLoss
from util.metric.registration_metric import folds_count_metric, folds_ratio_metric, mse_metric, jacobian_determinant
from util.metric.segmentation_metric import dice_metric
from util.visual.tensorboard_visual import tensorboard_visual_deformation
from util import loss


def train_reg(config, basedir):
    writer = SummaryWriter(log_dir=os.path.join(basedir, "logs"))
    model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 2))

    """
     ------------------------------------------------
               preparing dataset
     ------------------------------------------------
    """
    with open(config['TrainConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    train_dataset = PairDataset(dataset_config, 'train_pair')
    val_dataset = PairDataset(dataset_config, 'val_pair')
    test_dataset = PairDataset(dataset_config, 'test_pair')
    print(f'train dataset size: {len(train_dataset)}')
    print(f'val dataset size: {len(val_dataset)}')
    print(f'test dataset size: {len(test_dataset)}')
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
    reg_net = get_reg_model(config['ModelConfig'], checkpoint)
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
    loss_function_dict = get_loss_function(config)
    step = 0
    best_val_dice_metric = -1. * float('inf')

    for epoch in range(1, config["TrainConfig"]["epoch"] + 1):
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
            label1 = label1.to(device)
            volume2 = volume2.to(device)
            label2 = label2.to(device)

            optimizer.zero_grad()

            dvf = reg_net(volume1, volume2)

            warp_volume1 = STN_bilinear(volume1, dvf)
            # grid_sample_3d does not implement about int/uint8.
            warp_label1 = STN_nearest(label1.float(), dvf).type(torch.uint8)

            axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
            warp_label1_one_hot = F.one_hot(warp_label1.squeeze(dim=1).long()).permute(axis_order).contiguous()
            label2_one_hot = F.one_hot(label2.squeeze(dim=1).long()).permute(axis_order).contiguous()

            loss_dict = compute_reg_loss(config, dvf, loss_function_dict, warp_volume1, warp_label1_one_hot,
                                         volume2, label2_one_hot)

            loss_dict['total_loss'].backward()
            optimizer.step()
            update_dict(train_losses_dict, loss_dict)

            metric_dict = compute_reg_metric(dvf.clone().detach(), warp_volume1.clone().detach(),
                                             warp_label1_one_hot.clone().detach(),
                                             volume2.clone().detach(), label2_one_hot.clone().detach())

            update_dict(train_metrics_dict, metric_dict)
            step += 1
        mean_train_loss_dict = mean_dict(train_losses_dict)
        mean_train_metric_dict = mean_dict(train_metrics_dict)
        print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
        for loss_type, loss_value in mean_train_loss_dict.items():
            writer.add_scalar("train/loss/" + loss_type, loss_value, epoch)
        print(f"training total loss: {mean_train_loss_dict['total_loss']}")

        for metric_type, metric_value in mean_train_metric_dict.items():
            writer.add_scalar("train/metric/" + metric_type, metric_value, epoch)
        print(f"training dice: {mean_train_loss_dict['dice']}")
        writer.add_scalar("lr", optimizer.get_cur_lr(), epoch)
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
            for i, (id1, volume1, label1, id2, volume2, label2) in tqdm(enumerate(val_dataloader)):
                volume1 = volume1.to(device)
                label1 = label1.to(device)
                volume2 = volume2.to(device)
                label2 = label2.to(device)
                dvf = reg_net(volume1, volume2)
                warp_volume1 = STN_bilinear(volume1, dvf)
                warp_label1 = STN_nearest(label1.float(), dvf).type(torch.uint8)

                axis_order = (0, label1.dim() - 1) + tuple(range(1, label1.dim() - 1))
                warp_label1_one_hot = F.one_hot(warp_label1.squeeze(dim=1).long()).permute(axis_order).contiguous()
                label2_one_hot = F.one_hot(label2.squeeze(dim=1).long()).permute(axis_order).contiguous()

                loss_dict = compute_reg_loss(config, dvf, loss_function_dict, warp_volume1, warp_label1_one_hot,
                                             volume2, label2_one_hot)
                update_dict(val_losses_dict, loss_dict)

                metric_dict = compute_reg_metric(dvf.clone().detach(), warp_volume1.clone().detach(),
                                                 warp_label1_one_hot.clone().detach(),
                                                 volume2.clone().detach(), label2_one_hot.clone().detach())
                update_dict(val_metrics_dict, metric_dict)

                tensorboard_visual_registration(mode='val', name=id1[0] + '_' + id2[0] + '/img', writer=writer,
                                                step=epoch, fix=volume2[0][0].detach().cpu(),
                                                mov=volume1[0][0].detach().cpu(),
                                                reg=warp_volume1[0][0].detach().cpu())
                tensorboard_visual_registration(mode='val', name=id1[0] + '_' + id2[0] + '/seg', writer=writer,
                                                step=epoch, fix=label2[0][0].detach().cpu(),
                                                mov=label1[0][0].detach().cpu(), reg=warp_label1[0][0].detach().cpu())

                tag = 'val/' + id1[0] + '_' + id2[0]

                tensorboard_visual_deformation(name=tag + '/deformation', dvf=dvf[0].detach().cpu(),
                                               grid_spacing=dvf.shape[-1] // 30, writer=writer, step=epoch,
                                               linewidth=1, color='darkblue')

                tensorboard_visual_dvf(name=tag + '/dvf', dvf=dvf[0].detach().cpu(), writer=writer, step=epoch)

                det = jacobian_determinant(dvf[0].detach().cpu())

                tensorboard_visual_det(name=tag + '/det', det=det, writer=writer, step=epoch, normalize_by='slice',
                                       threshold=0, cmap='gray')

            mean_val_loss_dict = mean_dict(val_losses_dict)
            mean_val_metric_dict = mean_dict(val_metrics_dict)

            print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
            for loss_type, loss_value in mean_val_loss_dict.items():
                writer.add_scalar("val/loss/" + loss_type, loss_value, epoch)
            print(f"val total loss: {mean_val_loss_dict['dice']}")
            for metric_type, metric_value in mean_val_metric_dict.items():
                writer.add_scalar("val/metric/" + metric_type, metric_value, epoch)
            print(f"val dice: {mean_val_metric_dict['dice']}")

            if mean_val_metric_dict['dice'] > best_val_dice_metric:
                best_val_dice_metric = mean_val_metric_dict['dice']
                model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                 {"model": reg_net.state_dict(), "optim": optimizer.state_dict()})

    model_saver.save(
        os.path.join(basedir, "checkpoint", "last_epoch.pth"),
        {"model": reg_net.state_dict(), "optim": optimizer.state_dict()})


def compute_reg_loss(config, dvf, loss_function_dict, warp_volume1, warp_label1_one_hot, volume2, label2_one_hot):
    loss_dict = {}

    loss_dict['similarity_loss'] = loss_function_dict['similarity_loss'](warp_volume1, volume2)

    loss_dict['segmentation_loss'] = loss_function_dict['segmentation_loss'](warp_label1_one_hot, label2_one_hot)

    if config['LossConfig']['component']['gradient_loss']:
        loss_dict['gradient_loss'] = loss_function_dict['gradient_loss'](dvf)

    if config['LossConfig']['component']['bending_energy_loss']:
        loss_dict['bending_energy_loss'] = loss_function_dict['bending_energy_loss'](dvf)

    total_loss = 0.
    for loss_type in loss_dict.keys():
        total_loss = total_loss + config['LossConfig']['weight'][loss_type] * loss_dict[loss_type]
    loss_dict['total_loss'] = total_loss
    return loss_dict


def compute_reg_metric(dvf, warp_volume1, warp_label1, volume2, label2):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(warp_label1, label2).item()
    metric_dict['folds_ratio'] = folds_ratio_metric(dvf).item()
    metric_dict['folds_count'] = folds_count_metric(dvf).item()
    metric_dict['mse'] = mse_metric(warp_volume1, volume2).item()
    return metric_dict


def get_loss_function(config):
    loss_function_dict = dict()

    loss_function_dict['similarity_loss'] = getattr(loss, config['LossConfig']['component']['similarity_loss'])()
    loss_function_dict['segmentation_loss'] = getattr(loss, config['LossConfig']['component']['segmentation_loss'])()

    if config['LossConfig']['component']['gradient_loss']:
        loss_function_dict['gradient_loss'] = GradientLoss()
    if config['LossConfig']['component']['bending_energy_loss']:
        loss_function_dict['bending_energy_loss'] = BendingEnergyLoss()
    return loss_function_dict
