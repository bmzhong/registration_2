import json
import numpy as np


def generate_segmentation_available(M, dataset_config, output_path):
    train_data_names = np.array(dataset_config['train'])

    segmentation_available = {name: False for name in train_data_names}
    segmentation_available_true_list = np.random.choice(train_data_names, size=M, replace=False).tolist()
    segmentation_available_true_dict = {name: True for name in segmentation_available_true_list}
    segmentation_available.update(segmentation_available_true_dict)
    dataset_config['segmentation_available'] = segmentation_available
    dataset_config['M'] = M
    with open(output_path, 'w') as f:
        json.dump(dataset_config, f, indent=4)


if __name__ == '__main__':
    np.random.seed(0)

    data_path = '../../datasets/json/LPBA40.json'
    with open(data_path, 'r') as f:
        dataset_config = json.load(f)
    train_size = dataset_config['train_size']
    M_list = [0, 1, int(train_size * 0.5), int(train_size * 1.0)]
    for M in M_list:
        output_path = '../../datasets/json/LPBA40_' + str(M) + '.json'
        generate_segmentation_available(M=M, dataset_config=dataset_config, output_path=output_path)

    # data_path = '../../datasets/json/soma_nuclei.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, int(train_size * 0.5), int(train_size * 1.0)]
    # for M in M_list:
    #     output_path = '../../datasets/json/soma_nuclei_' + str(M) + '.json'
    #     generate_segmentation_available(M=M, dataset_config=dataset_config, output_path=output_path)


    # data_path = '../../datasets/json/OASIS.json'
    # with open(data_path, 'r') as f:
    #     dataset_config = json.load(f)
    # train_size = dataset_config['train_size']
    # M_list = [0, 1, int(train_size * 0.5), int(train_size * 1.0)]
    # for M in M_list:
    #     output_path = '../../datasets/json/OASIS_' + str(M) + '.json'
    #     generate_segmentation_available(M=M, dataset_config=dataset_config, output_path=output_path)