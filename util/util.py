from shutil import rmtree
from time import strftime, localtime
import torch
import os
import numpy as np
import random
import sys


def get_basedir(base_dir, start_new_model=False):
    # init the output folder structure
    if base_dir is None:
        base_dir = os.path.join("./", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    if start_new_model and os.path.exists(base_dir):
        rmtree(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(os.path.join(base_dir, 'logs')):
        os.mkdir(os.path.join(base_dir, 'logs'))  ##tensorboard
    if not os.path.exists(os.path.join(base_dir, 'checkpoint')):
        os.mkdir(os.path.join(base_dir, 'checkpoint'))  ##checkpoint
    return base_dir


def set_random_seed(seed=0):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def update_dict(total_dict: dict, each_dict: dict):
    for key, value in each_dict.items():

        if torch.is_tensor(value):
            value = value.item()
        if total_dict.get(key, None) is None:
            total_dict[key] = [value]
        else:
            total_dict[key].append(value)


def mean_dict(total_dict):
    mean_value = dict()
    for key, value in total_dict.items():
        value = np.array(value)
        value[np.isinf(value)] = 0
        mean_value[key] = np.mean(value)
    return mean_value


def std_dict(total_dict):
    std_value = dict()
    for key, value in total_dict.items():
        value = np.array(value)
        value[np.isinf(value)] = 0
        std_value[key] = np.std(value)
    return std_value


def swap_training(network_to_train, network_to_not_train):
    """
        Switch out of training one network and into training another
    """

    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
