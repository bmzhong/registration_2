import os

import monai
import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *

label_ids = [2, 3, 41, 42] + [4, 7, 8, 43, 46, 47]


def write_IXI(image_size, source_path, output_path, scale_factor):
    label_map = dict()
    for i, id in enumerate(label_ids):
        label_map[id] = i + 1

    if os.path.exists(output_path):
        os.remove(output_path)

    file = h5py.File(output_path, 'w')

    volume_resize = monai.transforms.Resize(spatial_size=image_size, mode='trilinear', align_corners=False)
    label_resize = monai.transforms.Resize(spatial_size=image_size, mode='nearest', align_corners=None)

    for dir_name in tqdm(os.listdir(source_path)):
        img_dir = os.path.join(source_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = 'volume.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = 'label.nii.gz'
            label_path = os.path.join(img_dir, label_name)

            volume = sitk.ReadImage(volume_path)
            volume = sitk.GetArrayFromImage(volume)
            volume = volume.astype(np.float64)
            volume = volume[np.newaxis, ...]
            volume = volume_resize(volume)
            volume = volume.squeeze(dim=0)
            volume = ((volume - volume.min()) / (volume.max() - volume.min())) * scale_factor

            volume = volume.astype(np.float32)

            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            label = label.astype(np.int32)
            new_label = np.zeros(label.shape, label.dtype)

            for origin_label, target_label in label_map.items():
                new_label[label == origin_label] = target_label
            label = new_label

            label_value = np.unique(label).tolist()
            print(label_value)

            label = label[np.newaxis, ...]
            label = label_resize(label)
            label = label.squeeze(dim=0)

            label.astype(np.uint8)
            image_group = file.create_group(dir_name)
            image_group.create_dataset(name='volume', data=volume)
            image_group.create_dataset(name='label', data=label)
    # file.attrs['label_map'] = [[origin_label, target_label] for origin_label, target_label in label_map.items()]

    file.attrs['image_size'] = image_size
    file.attrs['dataset_size'] = len(file.keys())
    file.attrs['region_number'] = len(label_map.values())
    file.attrs['normalize'] = [0, scale_factor]

    train_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\Train"
    val_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\Val"
    test_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\Test"


    train_names, val_names, test_names = [], [], []
    for path in os.listdir(train_path):
        train_names.append(os.path.basename(path).split(".")[0])

    for path in os.listdir(val_path):
        val_names.append(os.path.basename(path).split(".")[0])

    for path in os.listdir(test_path):
        test_names.append(os.path.basename(path).split(".")[0])

    file.attrs['train'] = train_names
    file.attrs['val'] = val_names
    file.attrs['test'] = test_names

    file.close()


# def split_train_val_test(hdf5_path):
#     file = h5py.File(hdf5_path, 'a')
#     train_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\Train"
#     val_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\Val"
#     test_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\Test"
#
#     train_names, val_names, test_names = [], [], []
#
#     for path in os.listdir(train_path):
#         train_names.append(os.path.basename(path).split(".")[0])
#
#     for path in os.listdir(val_path):
#         val_names.append(os.path.basename(path).split(".")[0])
#
#     for path in os.listdir(test_path):
#         test_names.append(os.path.basename(path).split(".")[0])
#
#     file.attrs['train'] = train_names
#     file.attrs['val'] = val_names
#     file.attrs['test'] = test_names
#     file.close()


if __name__ == '__main__':
    hdf5_path = '../../datasets/hdf5/IXI.h5'
    # split_train_val_test(hdf5_path)
    # source_path = r'G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data_nii\subject'
    # output_path = '../../datasets/hdf5/IXI.h5'
    # image_size = [128, 128, 128]
    # scale_factor = 1.
    # write_IXI(image_size, source_path, output_path, scale_factor)
    # hdf5_path = '../../datasets/hdf5/IXI.h5'
    # output_dir = r'G:\biomdeical\registration\data\datasets'
    # extract_hdf5(hdf5_path, output_dir)
