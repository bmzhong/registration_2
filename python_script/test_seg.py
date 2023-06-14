import os
import time
from shutil import copyfile

import torch
import torch.nn.functional as F
from util.data_util.dataset import SingleDataset
from torch.utils.data import DataLoader
import json
from model.segmentation.model_util import get_seg_model
import monai
import numpy as np
from tqdm import tqdm
from util.metric.segmentation_metric import dice_metric, ASD_metric, HD_metric
from util.util import update_dict, mean_dict, std_dict
from util.visual.image_util import write_image, save_image_figure
import csv


def test_seg(config, basedir, checkpoint=None, model_config=None):
    csv_file = open(os.path.join(basedir, "result.csv"), 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([str(time.asctime())])
    header_names = ['dice', 'ASD', 'HD']
    csv_writer.writerow(["image"] + header_names)

    with open(config['TestConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)

    num_classes = dataset_config['region_number']
    test_dataset = SingleDataset(dataset_config, 'test')

    print(f'test dataset size: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if checkpoint is None:
        checkpoint = config['TestConfig']['checkpoint']

    copyfile(checkpoint, os.path.join(
        basedir, "checkpoint", "checkpoint.pth"))

    if model_config is None:
        model_config = config['ModelConfig']

    seg_net = get_seg_model(model_config, dataset_config['region_number'] + 1, checkpoint)

    total = sum([param.nelement() for param in seg_net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    print(f'gpu: {config["TestConfig"]["gpu"]}')
    device = torch.device("cuda:0" if len(config["TestConfig"]["gpu"]) > 0 else "cpu")

    gpu_num = len(config["TestConfig"]["gpu"].split(","))
    seg_net.to(device)
    if gpu_num > 1:
        seg_net = torch.nn.DataParallel(seg_net, device_ids=[i for i in range(gpu_num)])

    seg_net.eval()
    test_metrics = dict()
    with torch.no_grad():
        for id, volume, label in tqdm(test_loader):
            volume = volume.to(device)
            label = label.to(device)

            predict = seg_net(volume)

            predict_softmax = F.softmax(predict, dim=1)

            axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
            label_one_hot = F.one_hot(label.squeeze(dim=1).long(), num_classes=num_classes + 1).permute(
                axis_order).contiguous()

            predict_one_hot = F.one_hot(torch.argmax(predict, dim=1).long(), num_classes=num_classes + 1).permute(
                axis_order).contiguous()

            metric_dict = compute_seg_metric(predict_one_hot, label_one_hot)

            update_dict(test_metrics, metric_dict)
            row_list = [id[0]]
            for metric_name in header_names:
                row_list.append(f"{metric_dict[metric_name]:.6f}")
            csv_writer.writerow(row_list)

            if config["TestConfig"]["save_image"]:
                predict_argmax = torch.argmax(predict_softmax, dim=1, keepdim=True)
                output_dir = os.path.join(basedir, "images", id[0])
                write_image(output_dir, id[0], predict_argmax[0][0], 'label')
                save_image_figure(output_dir, id[0] + '_image_slice', volume[0][0].detach().cpu())
                save_image_figure(output_dir, id[0] + '_label_slice', label[0][0].detach().cpu())
                save_image_figure(output_dir, id[0] + '_predict_slice', predict_argmax[0][0].detach().cpu())

    mean_metric = mean_dict(test_metrics)
    std_metric = std_dict(test_metrics)

    row_list = ["mean"]
    for metric_name in header_names:
        row_list.append(f"{mean_metric[metric_name]:.6f}")
    csv_writer.writerow(row_list)

    row_list = ["std"]
    for metric_name in header_names:
        row_list.append(f"{std_metric[metric_name]:.6f}")
    csv_writer.writerow(row_list)

    for key in mean_metric.keys():
        print(f"mean {key}: {mean_metric[key]}, std {key}: {std_metric[key]}")

    csv_file.close()


def compute_seg_metric(predict, target):
    metric_dict = dict()
    metric_dict['dice'] = dice_metric(predict, target).item()
    metric_dict['ASD'] = ASD_metric(predict, target).item()
    metric_dict['HD'] = HD_metric(predict, target).item()
    return metric_dict
