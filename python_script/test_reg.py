import os
from shutil import copyfile

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.registration.SpatialNetwork import SpatialTransformer
from model.registration.model_util import get_reg_model
from util.data_util.dataset import PairDataset, InferPairDataset
from torch.utils.data import DataLoader
import json

from util.metric.registration_metric import jacobian_determinant, ssim_metric, folds_percent_metric, SDLogJ_metric, \
    mse_metric
from util.metric.segmentation_metric import dice_metric, ASD_metric, HD_metric, dice_metric2, ASD_metric2, HD_metric2
from util.util import update_dict, mean_dict, std_dict
from util.visual.image_util import write_image, save_image_figure, save_deformation_figure, save_det_figure, \
    save_dvf_figure, save_RGB_dvf_figure, save_RGB_deformation_2_figure, save_warp_grid_figure
import time
import csv

from util.visual.visual_registration import mk_grid_img


def test_reg(config, basedir, checkpoint=None, model_config=None):
    print(f"start test time: {time.asctime()}")
    csv_file = open(os.path.join(basedir, "result.csv"), 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)

    # csv_writer.writerow([str(time.asctime())])

    # header_names = ['dice', 'ASD', 'HD', 'SSIM', 'folds_percent', 'SDLogJ', 'MSE']
    header_names = ['dice','SSIM', 'folds_percent']
    csv_writer.writerow(["moving_fixed"] + header_names)

    with open(config['TestConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    num_classes = dataset_config['region_number']
    if dataset_config['registration_type'] == 1 or dataset_config['registration_type'] == 2:
        atlas = dataset_config['atlas']
    else:
        atlas = None
    test_dataset = InferPairDataset(dataset_config, dataset_type='test',
                                    registration_type=dataset_config['registration_type'], atlas=atlas)
    print(f'test dataset size: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if checkpoint is None:
        checkpoint = config['TestConfig']['checkpoint']

    copyfile(checkpoint, os.path.join(
        basedir, "checkpoint", "checkpoint.pth"))

    if model_config is None:
        model_config = config['ModelConfig']
    reg_net = get_reg_model(model_config, checkpoint, dataset_config['image_size'])

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
    raw_grid_img = mk_grid_img(4, 1, dataset_config['image_size']).to(device)
    reg_net.eval()
    test_metrics_dict = dict()
    with torch.no_grad():
        for id1, volume1, label1, id2, volume2, label2 in tqdm(test_loader):
            volume1 = volume1.to(device)
            label1 = label1.to(device)
            volume2 = volume2.to(device)
            label2 = label2.to(device)

            dvf = reg_net(volume1, volume2)

            warp_volume1 = STN_bilinear(volume1, dvf)
            warp_label1 = STN_nearest(label1.float(), dvf).type(torch.uint8)

            metric_dict = compute_reg_metric(dvf, warp_volume1, warp_label1, volume2, label2,
                                             dataset_config['region_number'])
            update_dict(test_metrics_dict, metric_dict)
            grid_img = raw_grid_img.repeat(dvf.shape[0], 1, 1, 1, 1)
            warp_grid = STN_bilinear(grid_img.float(), dvf)

            row_list = [id1[0] + "_" + id2[0]]
            for metric_name in header_names:
                row_list.append(f"{metric_dict[metric_name]:.6f}")
            # csv_writer.writerow(row_list)

            if config["TestConfig"]["save_image"]:
                output_dir = os.path.join(basedir, "images", id1[0])
                write_image(output_dir, id1[0] + '(warped)_' + id2[0], warp_volume1[0][0].detach().cpu(), 'volume')
                write_image(output_dir, id1[0] + '(warped)_' + id2[0], warp_label1[0][0].detach().cpu(), 'label')
                flow = dvf[0].detach().cpu().permute(1, 2, 3, 0).contiguous()
                write_image(output_dir, id1[0] + '_' + id2[0], flow, 'flow')
                tag = ' ( ' + id1[0] + '_' + id2[0] + ' )'
                save_image_figure(output_dir, 'mov_image' + tag, volume1[0][0].detach().cpu(), normalize_by='volume')
                save_image_figure(output_dir, 'mov_label' + tag, label1[0][0].detach().cpu(), normalize_by='slice')
                save_image_figure(output_dir, 'fix_image' + tag, volume2[0][0].detach().cpu(), normalize_by='volume')
                save_image_figure(output_dir, 'fix_label' + tag, label2[0][0].detach().cpu(), normalize_by='slice')
                save_image_figure(output_dir, 'reg_image' + tag, warp_volume1[0][0].detach().cpu(),
                                  normalize_by='volume')
                save_image_figure(output_dir, 'reg_label' + tag, warp_label1[0][0].detach().cpu(), normalize_by='slice')
                save_deformation_figure(output_dir, 'deformation' + tag, dvf[0].detach().cpu(),
                                        grid_spacing=dvf.shape[-1] // 50, linewidth=1, color='darkblue')
                save_dvf_figure(output_dir, 'displacement_vector_field' + tag, dvf[0].detach().cpu())
                det = jacobian_determinant(dvf[0].detach().cpu())
                save_det_figure(output_dir, 'jacobian_determinant' + tag, det, normalize_by='slice', threshold=0,
                                cmap='Blues')
                save_RGB_dvf_figure(output_dir, 'rgb_dvf' + tag, dvf[0].detach().cpu())
                save_RGB_deformation_2_figure(output_dir, 'deformation_2' + tag, dvf[0].detach().cpu())
                save_warp_grid_figure(output_dir, 'warp_grid' + tag, warp_grid[0][0].detach().cpu())

        mean_test_metric_dict = mean_dict(test_metrics_dict)
        std_metric_dict = std_dict(test_metrics_dict)

        row_list = ["mean"]
        for metric_name in header_names:
            row_list.append(f"{mean_test_metric_dict[metric_name]:.6f}")
        csv_writer.writerow(row_list)

        row_list = ["std"]
        for metric_name in header_names:
            row_list.append(f"{std_metric_dict[metric_name]:.6f}")
        csv_writer.writerow(row_list)

        for key in mean_test_metric_dict.keys():
            print(f"mean {key}: {mean_test_metric_dict[key]}, std {key}: {std_metric_dict[key]}")

        csv_file.close()
    print(f"end test time: {time.asctime()}")

def compute_reg_metric(dvf, warp_volume1, warp_label1, volume2, label2, num_classes):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['ASD'] = ASD_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['HD'] = HD_metric2(warp_label1, label2, num_classes).item()
    metric_dict['SSIM'] = ssim_metric(warp_volume1, volume2)
    metric_dict['folds_percent'] = folds_percent_metric(dvf).item()
    # metric_dict['SDLogJ'] = SDLogJ_metric(dvf).item()
    # metric_dict['MSE'] = mse_metric(warp_volume1, volume2).item()

    return metric_dict
