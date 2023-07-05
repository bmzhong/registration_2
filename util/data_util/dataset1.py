import numpy as np
from torch.utils.data import Dataset
from util.data_util.file_util import HDF5Reader
import torch


class DynamicPairDataset(Dataset):
    def __init__(self):
        super(DynamicPairDataset, self).__init__()
        self.ids: list = []
        self.data: dict = {}

    def __getitem__(self, index):
        id = self.ids[index]
        img = self.data[id]
        return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2'], img['dvf']

    def update(self, data_dict: dict):
        self.ids = list(data_dict.keys())
        self.data = data_dict

    def __len__(self):
        return len(self.ids)
