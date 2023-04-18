import json

import monai
import numpy as np
from torch.utils.data import Dataset
from util.data_util.file_util import HDF5Reader
import torch
import matplotlib.pyplot as plt


class SingleDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None, data_names=None,
                 segmentation_available: dict = None):
        super(SingleDataset, self).__init__()
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.transform = transform
        self.data_names = dataset_config[dataset_type] if data_names is None else data_names
        self.segmentation_available = segmentation_available

    def __getitem__(self, index):
        img = self.file[self.data_names[index]]

        if self.segmentation_available is not None and self.segmentation_available.get(self.data_names[index],
                                                                                       True) is False:
            img['label'] = None

        img = self.as_type_to_tensor(img)
        if self.transform is None:
            return img['id'], img['volume'], img.get('label', None)
        img['volume'] = self.transform(img['volume'])
        return img['id'], img['volume'], img.get('label', None)

    def __len__(self):
        return len(self.data_names)

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.tensor(img[key])
        return img


class PairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None, data_names=None,
                 segmentation_available: dict = None):
        super(PairDataset, self).__init__()
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.transform = transform
        self.data_names = dataset_config[dataset_type] if data_names is None else data_names
        self.segmentation_available = segmentation_available

    def __getitem__(self, index):
        img1_name, img2_name = self.data_names[index]

        img1 = self.file[img1_name]

        if self.segmentation_available is not None and self.segmentation_available.get(img1_name, True) is False:
            img1['label'] = None

        img2 = self.file[img2_name]

        if self.segmentation_available is not None and self.segmentation_available.get(img2_name, True) is False:
            img2['label'] = None

        img = {'id1': img1['id'],
               'volume1': img1['volume'],
               'label1': img1.get('label', None),
               'id2': img2['id'],
               'volume2': img2['volume'],
               'label2': img2.get('label', None)}

        img = self.as_type_to_tensor(img)
        if self.transform is None:
            return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']
        img['volume1'] = self.transform(img['volume1'])
        img['volume2'] = self.transform(img['volume2'])
        return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

    def __len__(self):
        return len(self.data_names)

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.tensor(img[key])
        return img


if __name__ == '__main__':
    # json_path = '../../datasets/json/LPBA40.json'
    # with open(json_path, 'r') as f:
    #     dataset_config = json.load(f)
    # dataset = PairDataset(dataset_config, 'train_pair')
    # for i in range(len(dataset)):
    #     img = dataset[i]
    #     volume1 = img['volume1'][img['volume1'].shape[0] // 2, :, :]
    #     label1 = img['label1'][img['label1'].shape[0] // 2, :, :]
    #     volume2 = img['volume2'][img['volume2'].shape[0] // 2, :, :]
    #     label2 = img['label2'][img['label2'].shape[0] // 2, :, :]
    #     fig, ax = plt.subplots(2, 2)
    #     ax[0, 0].imshow(volume1, cmap='gray')
    #     ax[0, 1].imshow(label1, cmap='gray')
    #     ax[1, 0].imshow(volume2, cmap='gray')
    #     ax[1, 1].imshow(label2, cmap='gray')
    #     plt.show()
    #     print(len(img))
    #     break
    json_path = '../../datasets/json/LPBA40.json'
    with open(json_path, 'r') as f:
        dataset_config = json.load(f)
    dataset = SingleDataset(dataset_config, 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for id, volume, label in dataloader:
        print('h')

    # json_path = '../../datasets/json/LPBA40.json'
    # with open(json_path, 'r') as f:
    #     dataset_config = json.load(f)
    # dataset = PairDataset(dataset_config, 'train_pair')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    # for id1, volume1, label1, id2, volume2, label2 in dataloader:
    #     print('h')
