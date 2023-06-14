import csv
import json
import os
from datetime import time
from shutil import copyfile

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.mae.mae_model_util import get_mae_model
from util.ModelSaver import ModelSaver
from util.data_util.dataset import UnlabeledDataset
from util.Optimizer import Optimizer
import numpy as np

from util.visual.image_util import write_image, save_slice
from util.visual.tensorboard_visual import visual_gradient, tensorboard_visual_mae


def upsample_mask(x, mask, patch_size):
    N = x.shape[0]
    mask_shape = np.array(x.shape[2:]) // np.array(patch_size)
    mask = mask.reshape(N, *mask_shape).contiguous()
    mask = mask.unsqueeze(1)
    for i in range(len(x.shape[2:])):
        mask = mask.repeat_interleave(x.shape[2 + i] // mask.shape[2 + i], axis=2 + i)
    return mask


def pretrain_mae(config, basedir):
    print("train function: pretrain_mae")
    writer = SummaryWriter(log_dir=os.path.join(basedir, "logs"))
    model_saver = ModelSaver(config["TrainConfig"].get("max_save_num", 2))
    """
     ------------------------------------------------
               preparing dataset
     ------------------------------------------------
    """
    with open(config['TrainConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    train_dataset = UnlabeledDataset(dataset_config, dataset_type='train')
    val_dataset = UnlabeledDataset(dataset_config, dataset_type='val')
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
    model = get_mae_model(config['ModelConfig'], dataset_config['image_size'], checkpoint)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    print(f'gpu: {config["TrainConfig"]["gpu"]}')

    device = torch.device("cuda:0" if len(config["TrainConfig"]["gpu"]) > 0 else "cpu")
    model.to(device)
    gpu_num = len(config["TrainConfig"]["gpu"].split(","))
    if gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])

    optimizer = Optimizer(config=config["OptimConfig"],
                          model=model,
                          checkpoint=config["TrainConfig"]["checkpoint"])
    step = 0
    best_val_loss = float('inf')
    start_epoch = torch.load(checkpoint).get("epoch", 1) if checkpoint is not None else 1
    for epoch in range(start_epoch, config["TrainConfig"]["epoch"] + 1):
        """
         ------------------------------------------------
                 training network
         ------------------------------------------------
        """

        model.train()

        train_loss_list = []
        for id, volume in tqdm(train_dataloader):
            volume = volume.to(device)
            optimizer.zero_grad()
            loss, pred, mask = model(volume)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            step += 1
        print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")

        mean_train_loss = np.mean(train_loss_list)
        print(f"train total loss: {mean_train_loss}")
        writer.add_scalar("train/loss", mean_train_loss, epoch)
        writer.add_scalar("lr", optimizer.get_cur_lr(), epoch)
        # visual_gradient(model, writer, epoch)
        """
         ------------------------------------------------
                validating network
         ------------------------------------------------
        """
        if epoch % config['TrainConfig']['val_interval'] == 0:
            model.eval()
            val_loss_list = []
            val_count = 0
            with torch.no_grad():
                for id, volume in tqdm(val_dataloader):
                    volume = volume.to(device)
                    loss, pred, mask = model(volume)
                    val_loss_list.append(loss.item())
                    val_count += 1
                    if val_count < 8:
                        if config['ModelConfig']['type'] == 'UNet':
                            mask = upsample_mask(volume, mask, model.patch_size)
                            mask = mask.type_as(volume)
                            mask_volume = volume * (1. - mask)
                        else:
                            pred = model.unpatchify(pred)
                            patchify_volume = model.patchify(volume)
                            mask_volume = patchify_volume * mask.unsqueeze(dim=-1)
                            mask_volume = model.unpatchify(mask_volume)
                        tensorboard_visual_mae(mode='val', name=id[0] + '/img', writer=writer, step=epoch,
                                               img=volume[0][0].detach().cpu(),
                                               mask_img=mask_volume[0][0].detach().cpu(),
                                               pred_img=pred[0][0].detach().cpu())
            mean_val_loss = np.mean(val_loss_list)
            print(f"Epoch {epoch}/{config['TrainConfig']['epoch']}:")
            writer.add_scalar("val/loss", mean_val_loss, epoch)
            print(f"val total loss: {mean_val_loss}")
            if best_val_loss > mean_val_loss:
                best_val_loss = mean_val_loss
                if gpu_num > 1:
                    model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                     {"epoch": epoch, "model": model.module.state_dict(),
                                      "optim": optimizer.state_dict()})
                else:
                    model_saver.save(os.path.join(basedir, "checkpoint", "best_epoch.pth"),
                                     {"epoch": epoch, "model": model.state_dict(),
                                      "optim": optimizer.state_dict()})
    if gpu_num > 1:
        model_saver.save(
            os.path.join(basedir, "checkpoint", "last_epoch.pth"),
            {"epoch": config["TrainConfig"]["epoch"], "model": model.module.state_dict(),
             "optim": optimizer.state_dict()})
    else:
        model_saver.save(
            os.path.join(basedir, "checkpoint", "last_epoch.pth"),
            {"epoch": config["TrainConfig"]["epoch"], "model": model.state_dict(),
             "optim": optimizer.state_dict()})


def test_pretrain_mae(config, basedir, checkpoint=None, model_config=None):
    csv_file = open(os.path.join(basedir, "result.csv"), 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    header_names = ['id', 'mse']
    csv_writer.writerow(header_names)
    with open(config['TestConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    test_dataset = UnlabeledDataset(dataset_config, dataset_type='test')
    print(f'train dataset size: {len(test_dataset)}')
    test_dataloader = DataLoader(test_dataset, config["TrainConfig"]['batchsize'], shuffle=False)
    if checkpoint is None:
        checkpoint = config['TestConfig']['checkpoint']

    copyfile(checkpoint, os.path.join(basedir, "checkpoint", "checkpoint.pth"))
    if model_config is None:
        model_config = config['ModelConfig']
    model = get_mae_model(model_config, dataset_config['image_size'], checkpoint)
    print(f'gpu: {config["TestConfig"]["gpu"]}')
    device = torch.device("cuda:0" if len(config["TestConfig"]["gpu"]) > 0 else "cpu")

    model.to(device)
    gpu_num = len(config["TestConfig"]["gpu"].split(","))
    if gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    model.eval()
    test_loss_list = []
    model.eval()
    test_count = 0
    with torch.no_grad():
        for id, volume in tqdm(test_dataloader):
            volume = volume.to(device)
            loss, pred, mask = model(volume)
            test_loss_list.append(loss.item())
            # mask = upsample_mask(volume, mask, model.patch_size)
            # mask = mask.type_as(volume)
            # mask_volume = volume * (1. - mask)
            if config['ModelConfig']['type'] == 'UNet':
                mask = upsample_mask(volume, mask, model.patch_size)
                mask = mask.type_as(volume)
                mask_volume = volume * (1. - mask)
            else:
                pred = model.unpatchify(pred)
                patchify_volume = model.patchify(volume)
                mask_volume = patchify_volume * mask.unsqueeze(dim=-1)
                mask_volume = model.unpatchify(mask_volume)
            csv_writer.writerow([id[0], f"{loss.item():.6f}"])
            test_count += 1
            if config["TestConfig"]["save_image"] and test_count < 8:
                output_dir = os.path.join(basedir, "images", id[0])
                write_image(output_dir, 'volume_' + id[0], volume[0][0].detach().cpu(), 'volume')
                write_image(output_dir, 'mask_volume_' + id[0], mask_volume[0][0].detach().cpu(), 'volume')
                write_image(output_dir, 'pred_volume_' + id[0], pred[0][0].detach().cpu(), 'volume')
                img_list = [volume[0][0].detach().cpu(), mask_volume[0][0].cpu(), pred[0][0].detach().cpu()]
                name_list = ['img', 'mask_img', 'pred_img']
                save_slice(output_dir, img_list, name_list, cmap='gray', figsize=(18, 18))
    mean_test_loss = np.mean(test_loss_list)
    std_test_loss = np.std(test_loss_list)
    csv_writer.writerow(["mean", f"{mean_test_loss:.6f}"])
    csv_writer.writerow(["std", f"{std_test_loss:.6f}"])
    print(f"mean: {mean_test_loss}")
    print(f"std: {std_test_loss}")
    csv_file.close()
