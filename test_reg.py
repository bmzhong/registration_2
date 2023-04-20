import os
from shutil import copyfile

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.registration.SpatialNetwork import SpatialTransformer
from model.registration.model_util import get_reg_model
from train_reg import compute_reg_metric
from util.data_util.dataset import PairDataset
from torch.utils.data import DataLoader
import json

from util.metric.registration_metric import jacobian_determinant
from util.util import update_dict, mean_dict, std_dict
from util.visual.image_util import write_image, save_image_figure, save_deformation_figure, save_det_figure
import time


def test_reg(config, basedir, checkpoint=None, model_config=None):
    outfile = open(os.path.join(basedir, "log.txt"), 'a', encoding='utf-8')
    outfile.write(f"\n--------------{time.asctime()}------------------\n")
    with open(config['TestConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    test_dataset = PairDataset(dataset_config, 'test_pair')
    print(f'test dataset size: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if checkpoint is None:
        checkpoint = config['TestConfig']['checkpoint']

    copyfile(checkpoint, os.path.join(
        basedir, "checkpoint", "checkpoint.pth"))

    if model_config is None:
        model_config = config['ModelConfig']
    reg_net = get_reg_model(model_config, checkpoint)

    STN_bilinear = SpatialTransformer(dataset_config['image_size'], mode='bilinear')
    STN_nearest = SpatialTransformer(dataset_config['image_size'], mode='nearest')

    print(f'gpu: {config["TestConfig"]["gpu"]}')
    device = torch.device("cuda:0" if len(config["TestConfig"]["gpu"]) > 0 else "cpu")
    reg_net.to(device)
    STN_bilinear.to(device)
    STN_nearest.to(device)

    gpu_num = len(config["TestConfig"]["gpu"].split(","))
    if gpu_num > 1:
        reg_net = torch.nn.DataParallel(reg_net, device_ids=[i for i in range(gpu_num)])
        STN_bilinear = torch.nn.DataParallel(STN_bilinear, device_ids=[i for i in range(gpu_num)])
        STN_nearest = torch.nn.DataParallel(STN_nearest, device_ids=[i for i in range(gpu_num)])

    reg_net.eval()
    test_metrics_dict = dict()
    for id1, volume1, label1, id2, volume2, label2 in tqdm(test_loader):
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
        metric_dict = compute_reg_metric(dvf.clone().detach(), warp_volume1.clone().detach(),
                                         warp_label1_one_hot.clone().detach(),
                                         volume2.clone().detach(), label2_one_hot.clone().detach())
        update_dict(test_metrics_dict, metric_dict)

        for key, value in metric_dict.items():
            outfile.write(f"{id1[0]}_{id2[0]} {key}: {value}\n")

        if config["TestConfig"]["save_image"]:
            output_dir = os.path.join(basedir, "images", id1[0])
            write_image(output_dir, id1[0] + '(warped)_' + id2[0], warp_volume1[0][0].detach().cpu(), 'volume')
            write_image(output_dir, id1[0] + '(warped)_' + id2[0], warp_label1[0][0].detach().cpu(), 'label')
            tag = ' ( ' + id1[0] + '_' + id2[0] + ' )'
            save_image_figure(output_dir, 'mov_image' + tag, volume1[0][0].detach().cpu())
            save_image_figure(output_dir, 'mov_label' + tag, label1[0][0].detach().cpu())
            save_image_figure(output_dir, 'fix_image' + tag, volume2[0][0].detach().cpu())
            save_image_figure(output_dir, 'fix_label' + tag, label2[0][0].detach().cpu())
            save_image_figure(output_dir, 'reg_image' + tag, warp_volume1[0][0].detach().cpu())
            save_image_figure(output_dir, 'reg_label' + tag, warp_label1[0][0].detach().cpu())
            save_deformation_figure(output_dir, 'deformation' + tag, dvf[0].detach().cpu(),
                                    grid_spacing=dvf.shape[-1] // 30, linewidth=1, color='darkblue')
            det = jacobian_determinant(dvf[0].detach().cpu())
            save_det_figure(output_dir, 'jacobian_determinant' + tag, det, normalize_by='slice', threshold=0,
                            cmap='gray')

    mean_test_metric_dict = mean_dict(test_metrics_dict)
    std_metric_dict = std_dict(test_metrics_dict)
    for key in mean_test_metric_dict.keys():
        outfile.write(f"mean {key}: {mean_test_metric_dict[key]}, std {key}: {std_metric_dict[key]}\n")
        print(f"mean {key}: {mean_test_metric_dict[key]}, std {key}: {std_metric_dict[key]}")
    outfile.close()
