import random

import h5py
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math


def pairs_image_show(pairs):
    unique_sample = np.unique(np.array(pairs).reshape(-1)).tolist()
    sample_id_map = {sample: i for i, sample in enumerate(unique_sample)}
    n = len(unique_sample)
    relation = np.zeros([n, n])
    for img1, img2 in pairs:
        relation[sample_id_map[img1], sample_id_map[img2]] = 1
    plt.imshow(relation)
    plt.show()


def random_sampling_pairs(pairs, sampling_ratio):
    output_pair = []
    unique_sample = []
    pairs_map = dict()
    for img1, img2 in pairs:
        if pairs_map.get(img1, None) is None:
            pairs_map[img1] = [[img1, img2]]
            unique_sample.append(img1)
        else:
            pairs_map[img1].append([img1, img2])
    sampling_number = int(len(pairs) * sampling_ratio)
    per_sampling = sampling_number // len(unique_sample)
    residue = sampling_number - per_sampling * len(unique_sample)
    for i, img1 in enumerate(unique_sample):
        value = np.array(pairs_map[img1])
        indices = [i for i in range(len(value))]
        if i < residue:
            size = per_sampling + 1
        else:
            size = per_sampling
        sampling_value_indices = np.random.choice(
            indices, size=size, replace=False)
        sampling_value = value[sampling_value_indices]
        output_pair.extend(sampling_value.tolist())
    # value = np.array(pairs_map[unique_sample[-1]])
    # residue_size = int(len(pairs) * sampling_ratio) - len(output_pair)
    # assert 0 < residue_size <= len(value), 'residue size error'
    # indices = [i for i in range(len(value))]
    # sampling_value_indices = np.random.choice(indices, size=residue_size, replace=False)
    # sampling_value = value[sampling_value_indices]
    # output_pair.extend(sampling_value.tolist())
    # assert len(output_pair) == int(len(pairs) * sampling_ratio), 'sampling size error'
    return output_pair


def write_json(hdf5_path, output_path, train_size, val_size, test_size, sampling_ratio):
    h5_file = h5py.File(hdf5_path, 'r')
    json_data = dict()
    json_data['dataset_path'] = hdf5_path[hdf5_path.find('datasets'):]
    json_data['dataset_size'] = int(h5_file.attrs['dataset_size'])
    json_data['image_size'] = h5_file.attrs['image_size'].tolist()
    json_data['normalize'] = h5_file.attrs['normalize'].tolist()
    json_data['region_number'] = int(h5_file.attrs['region_number'])
    json_data['atlas'] = ''
    # if 'label_map' in h5_file.attrs.keys():
    #     label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
    # else:
    #     label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
    # json_data['label_map'] = label_value_map
    image_names = np.array(list(h5_file.keys()))
    train, val_test = train_test_split(image_names, train_size=train_size)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size))

    train = train.tolist()
    val = val.tolist()
    test = test.tolist()

    train_pairs = [[img1, img2] for img1 in train for img2 in train if img1 != img2]
    val_pairs = [[img1, img2] for img1 in val for img2 in val if img1 != img2]
    test_pairs = [[img1, img2] for img1 in test for img2 in test if img1 != img2]

    if sampling_ratio < 1.0:
        train_pairs = random_sampling_pairs(train_pairs, sampling_ratio)
        val_pairs = random_sampling_pairs(val_pairs, sampling_ratio)
        test_pairs = random_sampling_pairs(test_pairs, sampling_ratio)

    # pairs_image_show(train_pairs)
    # pairs_image_show(val_pairs)
    # pairs_image_show(test_pairs)

    json_data['train_size'] = len(train)
    json_data['val_size'] = len(val)
    json_data['test_size'] = len(test)

    json_data['train_pair_size'] = len(train_pairs)
    json_data['val_pair_size'] = len(val_pairs)
    json_data['test_pair_size'] = len(test_pairs)

    json_data['train'] = train
    json_data['val'] = val
    json_data['test'] = test

    json_data['train_pair'] = train_pairs
    json_data['val_pair'] = val_pairs
    json_data['test_pair'] = test_pairs

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)


def write_soma_nuclei_seretonin_json(hdf5_path, output_path):
    h5_file = h5py.File(hdf5_path, 'r')
    json_data = dict()
    json_data['dataset_path'] = hdf5_path[hdf5_path.find('datasets'):]
    json_data['dataset_size'] = int(h5_file.attrs['dataset_size'])
    json_data['image_size'] = h5_file.attrs['image_size'].tolist()
    json_data['normalize'] = h5_file.attrs['normalize'].tolist()
    json_data['region_number'] = int(h5_file.attrs['region_number'])
    if 'label_map' in h5_file.attrs.keys():
        label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
    else:
        label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
    json_data['label_map'] = label_value_map
    image_names = list(h5_file.keys())
    image_names.remove('allen')
    train = image_names
    val = ['190312_488_LP70_ET50_Z08_HF0_17-26-21_new',
           '190313_488_LP70_ET50_Z08_HF0_01-04-17',
           '181208_15_21_38']
    test = ['181207_10_39_06',
            '180725_20180724C2_LEFT_488_100ET_20-02-09',
            '190313_488_LP70_ET50_Z08_HF0_01-04-17',
            '180921_O11_488_LEFT_16-07-16',
            '190312_488_LP70_ET50_Z08_HF0_17-26-21',
            '180724_20180723O2_LEFT_488-2_100ET_16-13-41',
            '181207_18_26_44',
            '181208_15_21_38',
            '190524_488_ET50_0HF_LP70_18-43-49']

    train_pairs = [[img, 'allen'] for img in train]
    val_pairs = [[img, 'allen'] for img in val]
    test_pairs = [[img, 'allen'] for img in test]
    train = train + ['allen']
    json_data['train_size'] = len(train)
    json_data['val_size'] = len(val)
    json_data['test_size'] = len(test)

    json_data['train_pairs_size'] = len(train_pairs)
    json_data['val_pairs_size'] = len(val_pairs)
    json_data['test_pairs_size'] = len(test_pairs)

    json_data['train'] = train
    json_data['val'] = val
    json_data['test'] = test

    json_data['train_pair'] = train_pairs
    json_data['val_pair'] = val_pairs
    json_data['test_pair'] = test_pairs

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)


def write_soma_nuclei_json(hdf5_path, output_path):
    h5_file = h5py.File(hdf5_path, 'r')
    json_data = dict()
    json_data['dataset_path'] = hdf5_path[hdf5_path.find('datasets'):]
    json_data['dataset_size'] = int(h5_file.attrs['dataset_size'])
    json_data['image_size'] = h5_file.attrs['image_size'].tolist()
    json_data['normalize'] = h5_file.attrs['normalize'].tolist()
    json_data['region_number'] = int(h5_file.attrs['region_number'])
    if 'label_map' in h5_file.attrs.keys():
        label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
    else:
        label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
    json_data['label_map'] = label_value_map
    image_names = list(h5_file.keys())
    image_names.remove('allen')
    train, val_test = train_test_split(np.array(image_names), train_size=28)
    val, test = train_test_split(val_test, test_size=2 / 3)
    train = train.tolist() + ['allen']
    val = val.tolist()
    test = test.tolist()
    train_pairs = [[img, 'allen'] for img in train]
    val_pairs = [[img, 'allen'] for img in val]
    test_pairs = [[img, 'allen'] for img in test]

    json_data['train_size'] = len(train)
    json_data['val_size'] = len(val)
    json_data['test_size'] = len(test)

    json_data['train_pairs_size'] = len(train_pairs)
    json_data['val_pairs_size'] = len(val_pairs)
    json_data['test_pairs_size'] = len(test_pairs)

    json_data['train'] = train
    json_data['val'] = val
    json_data['test'] = test

    json_data['train_pair'] = train_pairs
    json_data['val_pair'] = val_pairs
    json_data['test_pair'] = test_pairs

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)


def write_LPBA40_json(hdf5_path, output_path):
    h5_file = h5py.File(hdf5_path, 'r')
    json_data = dict()
    json_data['dataset_path'] = hdf5_path[hdf5_path.find('datasets'):]
    json_data['dataset_size'] = int(h5_file.attrs['dataset_size'])
    json_data['image_size'] = h5_file.attrs['image_size'].tolist()
    json_data['normalize'] = h5_file.attrs['normalize'].tolist()
    json_data['region_number'] = int(h5_file.attrs['region_number'])
    json_data['atlas'] = 'S01'
    # if 'label_map' in h5_file.attrs.keys():
    #     label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
    # else:
    #     label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
    # json_data['label_map'] = label_value_map
    image_names = list(h5_file.keys())
    image_names.remove('S01')
    train, val_test = train_test_split(np.array(image_names), train_size=28)
    val, test = train_test_split(val_test, test_size=2 / 3)
    train = train.tolist()
    val = val.tolist()
    test = test.tolist()
    train_pairs = [[img, 'S01'] for img in train]
    val_pairs = [[img, 'S01'] for img in val]
    test_pairs = [[img, 'S01'] for img in test]


    json_data['train_size'] = len(train)
    json_data['val_size'] = len(val)
    json_data['test_size'] = len(test)

    json_data['train_pairs_size'] = len(train_pairs)
    json_data['val_pairs_size'] = len(val_pairs)
    json_data['test_pairs_size'] = len(test_pairs)

    json_data['train'] = train
    json_data['val'] = val
    json_data['test'] = test

    json_data['train_pair'] = train_pairs
    json_data['val_pair'] = val_pairs
    json_data['test_pair'] = test_pairs

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)


def write_IXI_json(hdf5_path, output_path):
    h5_file = h5py.File(hdf5_path, 'r')
    json_data = dict()
    json_data['dataset_path'] = hdf5_path[hdf5_path.find('datasets'):]
    json_data['dataset_size'] = int(h5_file.attrs['dataset_size'])
    json_data['image_size'] = h5_file.attrs['image_size'].tolist()
    json_data['normalize'] = h5_file.attrs['normalize'].tolist()
    json_data['region_number'] = int(h5_file.attrs['region_number'])
    json_data['atlas'] = 'atlas'

    train = h5_file.attrs['train'].tolist()
    val = h5_file.attrs['val'].tolist()
    test = h5_file.attrs['test'].tolist()

    train.append('atlas')

    train_pair = [['atlas', name] for name in train]
    val_pair = [['atlas', name] for name in val]
    test_pair = [['atlas', name] for name in test]

    json_data['train_size'] = len(train)
    json_data['val_size'] = len(val)
    json_data['test_size'] = len(test)

    json_data['train_pairs_size'] = len(train_pair)
    json_data['val_pairs_size'] = len(val_pair)
    json_data['test_pairs_size'] = len(test_pair)

    json_data['train'] = train
    json_data['val'] = val
    json_data['test'] = test

    json_data['train_pair'] = train_pair
    json_data['val_pair'] = val_pair
    json_data['test_pair'] = test_pair

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    LPBA40_path = '../../datasets/hdf5/LPBA40.h5'
    LPBA40_output_path = '../../datasets/json/LPBA40.json'
    write_LPBA40_json(LPBA40_path, LPBA40_output_path)

    # Mindboggle101_path = '../../datasets/hdf5/Mindboggle101.h5'
    # Mindboggle101_output_path = '../../datasets/json/Mindboggle101.json'
    # write_json(Mindboggle101_path, Mindboggle101_output_path, train_size=0.7,
    #            val_size=0.1, test_size=0.2, sampling_ratio=0.1)

    # OASIS_path = '../../datasets/hdf5/OASIS.h5'
    # OASIS_output_path = '../../datasets/json/OASIS.json'
    # write_json(OASIS_path, OASIS_output_path, train_size=0.7,
    #            val_size=0.1, test_size=0.2, sampling_ratio=0.01)

    # LPBA40_path = '../../datasets/hdf5/LPBA40_test.h5'
    # LPBA40_output_path = '../../datasets/json/LPBA40_test.json'
    # write_json(LPBA40_path, LPBA40_output_path,train_size=0.7,
    #            val_size=0.1, test_size=0.2, sampling_ratio=1.)

    # IXI_path = '../../datasets/hdf5/IXI.h5'
    # IXI_output_path = '../../datasets/json/IXI.json'
    # write_IXI_json(IXI_path, IXI_output_path)

    # soma_nuclei_4_path = '../../datasets/hdf5/soma_nuclei.h5'
    # soma_nuclei_4_output_path = '../../datasets/json/soma_nuclei.json'
    # write_soma_nuclei_json(soma_nuclei_4_path, soma_nuclei_4_output_path)

    # seretonin_path = '../../datasets/hdf5/soma_nuclei_seretonin.h5'
    # seretonin_output_path = '../../datasets/json/soma_nuclei_seretonin.json'
    # write_soma_nuclei_seretonin_json(seretonin_path, seretonin_output_path)
