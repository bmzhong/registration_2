import numpy as np

import h5py
import SimpleITK as sitk
import os

import torch
# from torchvision.transforms import functional as F
from torchvision.transforms import Resize
from collections import Counter
from hdf5_utils import *


def change_allen_resolution(root_path, output_dir, down_scale):
    os.makedirs(output_dir, exist_ok=True)

    volume_path = os.path.join(root_path, 'allen.nii.gz')
    label_path = os.path.join(root_path, 'allen_label.nii.gz')

    volume = sitk.ReadImage(volume_path)
    volume = sitk.GetArrayFromImage(volume)
    volume = volume[::down_scale, ::down_scale, ::down_scale]
    volume = sitk.GetImageFromArray(volume)

    sitk.WriteImage(volume, os.path.join(output_dir, 'allen.nii.gz'))

    label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label)
    label = label[::down_scale, ::down_scale, ::down_scale]
    label = sitk.GetImageFromArray(label)

    sitk.WriteImage(label, os.path.join(output_dir, 'allen_label.nii.gz'))


def change_soma_nuclei_resolution(root_path, output_dir, down_scale):
    for dir_name in os.listdir(root_path):
        img_dir = os.path.join(root_path, dir_name)
        volume_path = os.path.join(img_dir, dir_name + '.nii.gz')
        volume = sitk.ReadImage(volume_path)
        D, H, W = volume.GetSize()
        print(volume.GetSize())
        image_size = (D // down_scale, H // down_scale, W // down_scale)
        volume = sitk.GetArrayFromImage(volume)
        volume = volume[::down_scale, ::down_scale, ::down_scale]
        volume = sitk.GetImageFromArray(volume)
        # volume = resize_image_itk(volume, image_size, sitk.sitkLinear)
        print(volume.GetSize())

        label_path = os.path.join(img_dir, dir_name + '_label.nii.gz')
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)
        label = label[::down_scale, ::down_scale, ::down_scale]
        label = sitk.GetImageFromArray(label)
        # label = resize_image_itk(label, image_size, sitk.sitkNearestNeighbor)
        output_path = os.path.join(output_dir, dir_name)
        os.makedirs(output_path, exist_ok=True)
        sitk.WriteImage(volume, os.path.join(output_path, dir_name + '.nii.gz'))
        sitk.WriteImage(label, os.path.join(output_path, dir_name + '_label.nii.gz'))


def merge_soma_nuclei_label(root_path, preprocess_dir):
    # root_path = r'G:\biomdeical\registration\data\soma_nuclei\data\fix\180706_RIGHT_488_40ET_18-05-05\180706_RIGHT_488_40ET_18-05-05.tiff'
    # root_path = r'G:\biomdeical\registration\data\soma_nuclei\data\fix'
    # preprocess_dir = r'G:\biomdeical\registration\data\preprocess\merge_label'
    constrain_names = ['bs', 'cbx', 'cp', 'csc', 'ctx', 'hpf']
    constrain_label_value_map = {'bs': 8, 'cbx': 9, 'cp': 6, 'csc': 7, 'ctx': 10, 'hpf': 5}
    for dir_name in os.listdir(root_path):
        img_dir = os.path.join(root_path, dir_name)
        if os.path.isdir(img_dir):
            volume_path = os.path.join(img_dir, dir_name + '.tiff')
            volume = sitk.ReadImage(volume_path)
            volume = sitk.GetArrayFromImage(volume)
            label = np.zeros(volume.shape, dtype=int)
            for constrain_name in constrain_names:
                constrain_path = os.path.join(img_dir, dir_name + '_' + constrain_name + '.tiff')
                constrain = sitk.ReadImage(constrain_path)
                constrain = sitk.GetArrayFromImage(constrain)
                assert len(np.unique(constrain)) == 2, 'label value error'
                label_value = np.unique(constrain)[-1]

                if label_value != constrain_label_value_map[constrain_name]:
                    print(f'{dir_name}: the label value of {constrain_name} is {label_value}, but it '
                          f'should be {constrain_label_value_map[constrain_name]}')
                    label_value = constrain_label_value_map[constrain_name]
                label[constrain > 0] = label_value

            print(np.unique(label.reshape(-1).tolist()))
            volume = sitk.GetImageFromArray(volume)
            label = sitk.GetImageFromArray(label)
            preprocess_out_path = os.path.join(preprocess_dir, dir_name)
            os.makedirs(preprocess_out_path, exist_ok=True)
            sitk.WriteImage(volume, os.path.join(preprocess_out_path, dir_name + '.nii.gz'))
            sitk.WriteImage(label, os.path.join(preprocess_out_path, dir_name + '_label.nii.gz'))
        if 'allen' in dir_name:
            break


def get_soma_nuclei_label_map():
    root_path = r'G:\biomdeical\registration\data\preprocess\merge_label\soma_nuclei\soma_' \
                r'nuclei\180706_RIGHT_488_40ET_18-05-05\180706_RIGHT_488_40ET_18-05-05_label.nii.gz'
    label = sitk.ReadImage(root_path)
    label = sitk.GetArrayFromImage(label)
    label_unique = list(np.unique(label))
    label_unique.sort()
    label_map = dict()
    for i in range(len(label_unique)):
        label_map[label_unique[i]] = i
    print(label_map)
    return label_map


def write_soma_nuclei(scale_down):
    file = h5py.File('../../datasets/hdf5/soma_nuclei_' + str(scale_down) + '.h5', 'w')
    root_path = r'G:\biomdeical\registration\data\preprocess\merge_label\soma_nuclei\soma_nuclei_' + str(scale_down)
    allen_path = r'G:\biomdeical\registration\data\preprocess\merge_label\allen\allen_' + str(scale_down)
    scale_factor = 255.
    label_map = get_soma_nuclei_label_map()
    if scale_down == 2:
        image_size = [256, 224, 160]
    elif scale_down == 4:
        image_size = [128, 128, 128]
    for dir_name in ['allen'] + os.listdir(root_path):
        if dir_name == 'allen':
            img_dir = allen_path
        else:
            img_dir = os.path.join(root_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = dir_name + '.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = dir_name + '_label.nii.gz'
            label_path = os.path.join(img_dir, label_name)

            volume = sitk.ReadImage(volume_path)
            # if image_size is None:
            #     image_size = list(volume.GetSize())
            volume = resize_image_itk(volume, image_size, sitk.sitkLinear)
            volume = sitk.GetArrayFromImage(volume)
            volume = volume.astype(np.float64)
            volume = ((volume - volume.min()) / (volume.max() - volume.min())) * scale_factor
            volume = volume.astype(np.float32)

            label = sitk.ReadImage(label_path)
            label = resize_image_itk(label, image_size, sitk.sitkLinear)
            label = sitk.GetArrayFromImage(label)
            assert len(np.unique(label)) == len(label_map), 'label number error'
            for origin_label, target_label in label_map.items():
                label[label == origin_label] = target_label
            label.astype(np.uint8)

            image_group = file.create_group(dir_name)
            image_group.create_dataset(name='volume', data=volume)
            image_group.create_dataset(name='label', data=label)
    file.attrs['label_map'] = [[origin_label, target_label] for origin_label, target_label in label_map.items()]
    file.attrs['image_size'] = image_size
    file.attrs['dataset_size'] = len(file.keys())
    file.attrs['region_number'] = len(label_map) - 1
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':
    # root_path = r'G:\biomdeical\registration\data\soma_nuclei\data\moving'
    # preprocess_dir = r'G:\biomdeical\registration\data\preprocess\merge_label\allen'
    # merge_soma_nuclei_label(root_path, preprocess_dir)

    # root_path = r'G:\biomdeical\registration\data\soma_nuclei\data\fix'
    # preprocess_dir = r'G:\biomdeical\registration\data\preprocess\merge_label\soma_nuclei\soma_nuclei'
    # merge_soma_nuclei_label(root_path, preprocess_dir)

    # scale_factor = 4
    # root_path = r'G:\biomdeical\registration\data\preprocess\merge_label\soma_nuclei\soma_nuclei'
    # output_dir = r'G:\biomdeical\registration\data\preprocess\merge_label\soma_nuclei\soma_nuclei_' + str(scale_factor)
    # change_soma_nuclei_resolution(root_path, output_dir, scale_factor)

    # scale_factor = 4
    # root_path = r'G:\biomdeical\registration\data\preprocess\merge_label\allen\allen'
    # output_dir = r'G:\biomdeical\registration\data\preprocess\merge_label\allen\allen_' + str(scale_factor)
    # change_allen_resolution(root_path, output_dir, scale_factor)

    # label_map = get_soma_nuclei_label_map()
    scale_down = 4
    # write_soma_nuclei(scale_down=scale_down)

    hdf5_path = '../../datasets/hdf5/soma_nuclei_' + str(scale_down) + '.h5'
    output_dir = r'G:\biomdeical\registration\data\datasets'
    extract_hdf5(hdf5_path, output_dir)
