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
    if 'label_map' in h5_file.attrs.keys():
        label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
    else:
        label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
    json_data['label_map'] = label_value_map
    image_names = np.array(list(h5_file.keys()))
    train, val_test = train_test_split(image_names, train_size=train_size)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size))

    train = train.tolist()
    val = val.tolist()
    test = test.tolist()

    train_pairs = [[img1, img2] if img1 != img2 else None for img1 in train for img2 in train]
    val_pairs = [[img1, img2] if img1 != img2 else None for img1 in val for img2 in val]
    test_pairs = [[img1, img2] if img1 != img2 else None for img1 in test for img2 in test]
    train_pairs = list(filter(None, train_pairs))
    val_pairs = list(filter(None, val_pairs))
    test_pairs = list(filter(None, test_pairs))
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
    if 'label_map' in h5_file.attrs.keys():
        label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
    else:
        label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
    json_data['label_map'] = label_value_map
    image_names = list(h5_file.keys())
    image_names.remove('S01')
    train, val_test = train_test_split(np.array(image_names), train_size=28)
    val, test = train_test_split(val_test, test_size=2 / 3)
    train = train.tolist() + ['S01']
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

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # LPBA40_path = '../../datasets/hdf5/LPBA40.h5'
    # LPBA40_output_path = '../../datasets/json/LPBA40.json'
    # write_json(LPBA40_path, LPBA40_output_path, train_size=0.7,
    #            val_size=0.1, test_size=0.2, sampling_ratio=1.)

    # Mindboggle101_path = '../../datasets/hdf5/Mindboggle101.h5'
    # Mindboggle101_output_path = '../../datasets/json/Mindboggle101.json'
    # write_json(Mindboggle101_path, Mindboggle101_output_path, train_size=0.7,
    #            val_size=0.1, test_size=0.2, sampling_ratio=0.1)

    # OASIS_1_path = '../../datasets/hdf5/OASIS.h5'
    # OASIS_1_output_path = '../../datasets/json/OASIS.json'
    # write_json(OASIS_1_path, OASIS_1_output_path, train_size=0.7,
    #            val_size=0.1, test_size=0.2, sampling_ratio=0.01)

    # soma_nuclei_4_path = '../../datasets/hdf5/soma_nuclei.h5'
    # soma_nuclei_4_output_path = '../../datasets/json/soma_nuclei.json'
    # write_soma_nuclei_json(soma_nuclei_4_path, soma_nuclei_4_output_path)

    LPBA40_path = '../../datasets/hdf5/LPBA40.h5'
    LPBA40_output_path = '../../datasets/json/LPBA40.json'
    write_LPBA40_json(LPBA40_path, LPBA40_output_path)
