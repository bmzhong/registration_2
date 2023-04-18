import logging
import os
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
from util.metric.segmentation_metric import dice_metric
from util.visual.image_visual import write_image


def test(config, basedir, checkpoint=None, test_dataset=None):
    outfile = open(os.path.join(basedir, "logs", "log.txt"), 'w', encoding='utf-8')

    with open(config['TestConfig']['data_path'], 'r') as f:
        dataset_config = json.load(f)
    if test_dataset is None:
        test_dataset = SingleDataset(dataset_config, 'test')
    print(f'test dataset size: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if checkpoint is None:
        checkpoint = config['TestConfig']['checkpoint']

    copyfile(checkpoint, os.path.join(
        basedir, "checkpoint", "checkpoint.pth"))

    seg_net = get_seg_model(config['ModelConfig'], dataset_config['region_number'] + 1, checkpoint)

    total = sum([param.nelement() for param in seg_net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    print(f'gpu: {config["TestConfig"]["gpu"]}')
    device = torch.device("cuda:0" if len(config["TestConfig"]["gpu"]) > 0 else "cpu")

    gpu_num = len(config["TestConfig"]["gpu"].split(","))
    seg_net.to(device)
    if gpu_num > 1:
        seg_net = torch.nn.DataParallel(seg_net, device_ids=[i for i in range(gpu_num)])

    seg_net.eval()
    test_metrics = []
    for id, volume, label in tqdm(test_loader):
        volume = volume.to(device)
        label = label.to(device)

        predict = seg_net(volume)

        predict_softmax = F.softmax(predict, dim=1)

        axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
        label_one_hot = F.one_hot(label.squeeze(dim=1).long()).permute(axis_order).contiguous()

        predict_one_hot = F.one_hot(torch.argmax(predict, dim=1).long()).permute(axis_order).contiguous()

        dice = dice_metric(predict_one_hot, label_one_hot)

        test_metrics.append(dice.item())
        outfile.write(f'{id[0]}  dice: {dice} \n')
        if config["TestConfig"]["save_image"]:
            predict_argmax = torch.argmax(predict_softmax, dim=1, keepdim=True)
            output_dir = os.path.join(basedir, "images")
            write_image(output_dir, id[0], predict_argmax[0], 'label')

    mean_metric = np.mean(test_metrics)
    std_metric = np.std(test_metrics)
    outfile.write(f"mean test dice: {mean_metric}, std test dice: {std_metric} \n")
    outfile.close()
    print(f"mean test dice: {mean_metric}, std test dice: {std_metric}")
