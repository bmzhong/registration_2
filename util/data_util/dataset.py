import json

import monai
import numpy as np
from torch.utils.data import Dataset
from util.data_util.file_util import HDF5Reader
import torch
import matplotlib.pyplot as plt


class UnlabeledDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None, data_names=None):
        super(UnlabeledDataset, self).__init__()
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.transform = transform
        self.data_names = dataset_config[dataset_type] if data_names is None else data_names

    def __getitem__(self, index):
        img = self.file[self.data_names[index]]
        id = img['id']
        volume = img['volume']
        if self.transform is not None:
            volume = self.transform(volume)
        return id, volume

    def __len__(self):
        return len(self.data_names)


class SingleDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None,
                 seg=None, data_names=None):
        super(SingleDataset, self).__init__()
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.transform = transform
        self.data_names = dataset_config[dataset_type] if data_names is None else data_names
        self.seg = seg

    def __getitem__(self, index):
        img = self.file[self.data_names[index]]

        if self.seg is not None and self.seg.get(self.data_names[index],
                                                 True) is False:
            img['label'] = []

        img = self.as_type_to_tensor(img)

        if self.transform is None:
            return img['id'], img['volume'], img.get('label', [])

        img = self.process_transform(img)

        return img['id'], img['volume'], img.get('label', [])

    def __len__(self):
        return len(self.data_names)

    def process_transform(self, img):
        if 'label' in img.keys() and img['label'] == []:
            img.pop('label')

        img = self.transform(img)

        if 'label' not in img.keys():
            img['label'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


class PairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, transform=None,
                 seg=None, data_names=None):
        super(PairDataset, self).__init__()
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.transform = transform
        self.data_names = dataset_config[dataset_type] if data_names is None else data_names
        self.seg = seg

    def __getitem__(self, index):
        img1_name, img2_name = self.data_names[index]

        img1 = self.file[img1_name]
        img2 = self.file[img2_name]

        if self.seg is not None and self.seg.get(img1_name, True) is False:
            img1['label'] = []

        if self.seg is not None and self.seg.get(img2_name, True) is False:
            img2['label'] = []

        img = {'id1': img1['id'],
               'volume1': img1['volume'],
               'label1': img1.get('label', []),
               'id2': img2['id'],
               'volume2': img2['volume'],
               'label2': img2.get('label', [])}

        img = self.as_type_to_tensor(img)
        if self.transform is None:
            return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

        img = self.process_transform(img)

        return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

    def __len__(self):
        return len(self.data_names)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


class RandomPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, registration_type, transform=None,
                 seg=None, atlas=False, data_names=None):
        super(RandomPairDataset, self).__init__()
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.registration_type = registration_type
        self.transform = transform
        self.data_names = dataset_config[dataset_type] if data_names is None else data_names
        self.seg = seg
        self.atlas = atlas

    def __getitem__(self, index):
        if self.registration_type == 0:
            img1_name = self.data_names[index]
            img2_name = np.random.choice(self.data_names)
        elif self.registration_type == 1:
            img1_name = self.data_names[index]
            img2_name = self.atlas
        elif self.registration_type == 2:
            img1_name = self.atlas
            img2_name = self.data_names[index]
        else:
            raise Exception("registration type error!")

        img1 = self.file[img1_name]
        img2 = self.file[img2_name]

        if self.seg is not None and self.seg.get(img1_name, True) is False:
            img1['label'] = []

        if self.seg is not None and self.seg.get(img2_name, True) is False:
            img2['label'] = []

        img = {'id1': img1['id'],
               'volume1': img1['volume'],
               'label1': img1.get('label', []),
               'id2': img2['id'],
               'volume2': img2['volume'],
               'label2': img2.get('label', [])}

        img = self.as_type_to_tensor(img)

        if self.transform is None:
            return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

        img = self.process_transform(img)

        return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

    def __len__(self):
        return len(self.data_names)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
        return img


class TestPairDataset(Dataset):
    def __init__(self, dataset_config, dataset_type, registration_type, transform=None,
                 seg=None, atlas=False, data_names=None):
        self.file = HDF5Reader(dataset_config['dataset_path'])
        self.transform = transform
        self.seg = seg
        self.atlas = atlas
        data_names = dataset_config[dataset_type] if data_names is None else data_names
        if registration_type == 0:
            self.pair_data_names = [[name1, name2] for name1 in data_names for name2 in data_names if name1 != name2]
        elif registration_type == 1:
            self.pair_data_names = [[name1, self.atlas] for name1 in data_names]
        elif registration_type == 2:
            self.pair_data_names = [[self.atlas, name1] for name1 in data_names]
        else:
            raise Exception("registration type error!")
        if len(self.pair_data_names) > 200:
            indices = [i for i in range(len(self.pair_data_names))]
            indices = np.random.choice(indices, size=200, replace=False).tolist()
            self.pair_data_names = np.array(self.pair_data_names)[indices].tolist()

    def __getitem__(self, index):
        img1_name, img2_name = self.pair_data_names[index]

        img1 = self.file[img1_name]
        img2 = self.file[img2_name]

        if self.seg is not None and self.seg.get(img1_name, True) is False:
            img1['label'] = []

        if self.seg is not None and self.seg.get(img2_name, True) is False:
            img2['label'] = []

        img = {'id1': img1['id'],
               'volume1': img1['volume'],
               'label1': img1.get('label', []),
               'id2': img2['id'],
               'volume2': img2['volume'],
               'label2': img2.get('label', [])}

        img = self.as_type_to_tensor(img)
        if self.transform is None:
            return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

        img = self.process_transform(img)

        return img['id1'], img['volume1'], img['label1'], img['id2'], img['volume2'], img['label2']

    def __len__(self):
        return len(self.pair_data_names)

    def process_transform(self, img):
        if 'label1' in img.keys() and img['label1'] == []:
            img.pop('label1')
        if 'label2' in img.keys() and img['label2'] == []:
            img.pop('label2')

        img = self.transform(img)

        if 'label1' not in img.keys():
            img['label1'] = []
        if 'label2' not in img.keys():
            img['label2'] = []
        return img

    @staticmethod
    def as_type_to_tensor(img):
        for key, value in img.items():
            if isinstance(value, np.ndarray):
                img[key] = torch.from_numpy(img[key])
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
