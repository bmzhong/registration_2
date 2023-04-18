import h5py
import os
import json
import matplotlib.pyplot as plt
import numpy as np

class HDF5Reader:
    def __init__(self, path: str):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('\033[31m{} not found!\033[0m'.format(path))
            raise Exception

    def __getitem__(self, key):
        data = {'id': key}
        group = self.file[key]
        for k in group.keys():
            data[k] = np.expand_dims(group[k][:], axis=0)
        return data


if __name__ == '__main__':
    LPBA40_json_path = '../../datasets/json/LPBA40.json'
    with open(LPBA40_json_path, 'r') as f:
        LPBA40_json = json.load(f)
    file = HDF5Reader(LPBA40_json['dataset_path'])
    for img_name in LPBA40_json['train']:
        img_dict = file[img_name]
        img = img_dict['volume']
        label = img_dict['label']
        img_show = img[img.shape[0] // 2, :, :]
        label_show = label[label.shape[0] // 2, :, :]
        plt.imshow(img_show, cmap='gray')
        plt.show()
        plt.imshow(label_show, cmap='gray')
        plt.show()
        break
