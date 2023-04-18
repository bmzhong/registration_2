import json
import numpy as np


def LPBA40(M, data_path):
    with open(data_path, 'r') as f:
        dataset_config = json.load(f)

    train_data_names = np.array(dataset_config['train'])

    segmentation_available = {name: False for name in train_data_names}
    segmentation_available_true_list = np.random.choice(train_data_names, size=M, replace=False).tolist()
    segmentation_available_true_dict = {name: True for name in segmentation_available_true_list}
    segmentation_available.update(segmentation_available_true_dict)
    dataset_config['segmentation_available'] = segmentation_available
    dataset_config['M'] = M
    with open(data_path, 'w') as f:
        json.dump(dataset_config, f, indent=4)


if __name__ == '__main__':
    np.random.seed(0)
    data_path = '../../datasets/json/LPBA40.json'
    LPBA40(M=20, data_path=data_path)
