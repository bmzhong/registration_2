import json
import os
import random

import numpy as np


# def select_seg(M, dataset_config, output_path):
#
#     train_data_names = np.array(dataset_config['train'])
#
#     segmentation_available = {name: False for name in train_data_names}
#     segmentation_available_true_list = np.random.choice(train_data_names, size=M, replace=False).tolist()
#     segmentation_available_true_dict = {name: True for name in segmentation_available_true_list}
#     segmentation_available.update(segmentation_available_true_dict)
#     dataset_config['segmentation_available'] = segmentation_available
#     dataset_config['M'] = M
#     with open(output_path, 'w') as f:
#         json.dump(dataset_config, f, indent=4)

def select_seg(M_list, dataset_config, output_path):
    train_data_names = np.array(dataset_config['train'])
    for M in M_list:
        seg = {name: False for name in train_data_names}
        if dataset_config.get('atlas', '') != '':
            seg[dataset_config['atlas']] = False
        seg_true_list = np.random.choice(train_data_names, size=M, replace=False).tolist()
        if M > 1 and dataset_config.get('atlas', '') != '' and dataset_config['atlas'] not in seg_true_list:
            seg_true_list.append(dataset_config['atlas'])
        print(seg_true_list)
        seg_true_dict = {name: True for name in seg_true_list}
        seg.update(seg_true_dict)
        dataset_config['seg_' + str(M)] = seg
    with open(output_path, 'w') as f:
        json.dump(dataset_config, f, indent=4)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    data_path = '../../datasets/json/45_128_IXI_191.json'
    with open(data_path, 'r') as f:
        dataset_config = json.load(f)
    train_size = dataset_config['train_size']
    M_list = [0, 1, 5, int(train_size * 0.5), int(train_size * 1.0)]
    output_path = data_path
    select_seg(M_list=M_list, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/5_192_Mindboggle101.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, 5, int(train_size * 0.5), int(train_size * 1.0)]
    # output_path = data_path
    # select_seg(M_list=M_list, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/Mindboggle101.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1,5, int(train_size * 0.5), int(train_size * 1.0)]
    # output_path = data_path
    # select_seg(M_list=M_list, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/45_128_IXI.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, 5, int(train_size * 0.5), int(train_size * 1.0)]
    # output_path = data_path
    # select_seg(M_list=M_list, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/35_224_OASIS.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, 5, int(train_size * 0.5), int(train_size * 1.0)]
    # output_path = data_path
    # select_seg(M_list=M_list, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/soma_nuclei_seretonin.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # # M_list = [0, 1, int(train_size * 0.5), int(train_size * 1.0)]
    # M_list = [int(train_size * 1.0)]
    # for M in M_list:
    #     output_path = '../../datasets/json/soma_nuclei_seretonin_' + str(M) + '.json'
    #     select_seg(M=M, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/soma_nuclei.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, 5, int(train_size * 0.5), int(train_size * 1.0)]
    # output_path = data_path
    # select_seg(M_list=M_list, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/soma_nuclei.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, int(train_size * 0.5), int(train_size * 1.0)]
    # for M in M_list:
    #     output_path = '../../datasets/json/soma_nuclei_' + str(M) + '.json'
    #     generate_segmentation_available(M=M, dataset_config=dataset_config, output_path=output_path)
