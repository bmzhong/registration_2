# from torchvision.transforms import functional as F
import os

import monai
import numpy as np
from tqdm import tqdm

from preprocess.preprocess_hdf5.hdf5_utils import *


def get_LPBA40_label_map():
    root_path = r'G:\biomdeical\registration\public_data\LPBA40\LPBA40_Subjects_Delineation_Space_MRI_and_label_files\LPBA40subjects.delineation_space\LPBA40\delineation_space\S01\S01.delineation.structure.label.img.gz'
    label = sitk.ReadImage(root_path)

    label = sitk.GetArrayFromImage(label)
    label_unique = list(np.unique(label))
    label_unique.sort()
    label_map = dict()
    for i in range(len(label_unique)):
        label_map[label_unique[i]] = i
    print(label_map)
    # for origin_label, target_label in label_map.items():
    #     label[label == origin_label] = target_label
    # image = sitk.GetImageFromArray(label)
    # image =resize_image_itk(image,[160, 192, 160],sitk.sitkNearestNeighbor)
    # sitk.WriteImage(image, '../../datasets/temp_data/resample_S01_label_new.nii.gz')
    return label_map


def write_LPBA40():
    output_path = '../../datasets/hdf5/LPBA40.h5'
    if os.path.exists(output_path):
        os.remove(output_path)
    file = h5py.File(output_path, 'w')
    image_size = [32, 32, 32]
    root_path = r'G:\biomdeical\registration\public_data\LPBA40\LPBA40_Subjects_Delineation_Space_MRI_and' \
                r'_label_files\LPBA40subjects.delineation_space\LPBA40\delineation_space'

    # label_map = get_LPBA40_label_map()

    label_map = {0: 0,
                 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1,
                 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1,

                 41: 2, 42: 2, 43: 2, 44: 2, 45: 2, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2,

                 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3,

                 81: 4, 82: 4, 83: 4, 84: 4, 85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 4, 92: 4,

                 101: 5, 102: 5, 121: 5, 122: 5,

                 161: 6, 162: 6, 163: 6, 164: 6, 165: 6, 166: 6, 181: 6, 182: 6

                 }
    file.attrs['label_map'] = [[origin_label, target_label] for origin_label, target_label in label_map.items()]
    scale_factor = 255.
    volume_resize = monai.transforms.Resize(spatial_size=image_size, mode='trilinear', align_corners=False)
    label_resize = monai.transforms.Resize(spatial_size=image_size, mode='nearest', align_corners=None)
    for dir_name in tqdm(os.listdir(root_path)):
        img_dir = os.path.join(root_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = dir_name + '.delineation.skullstripped.img.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = dir_name + '.delineation.structure.label.img.gz'
            label_path = os.path.join(img_dir, label_name)

            volume = sitk.ReadImage(volume_path)
            # volume = resize_image_itk(volume, image_size, sitk.sitkLinear)
            volume = sitk.GetArrayFromImage(volume)
            volume = volume.astype(np.float64)
            volume = volume[np.newaxis, ...]
            volume = volume_resize(volume)
            volume = volume.squeeze(dim=0)
            volume = ((volume - volume.min()) / (volume.max() - volume.min())) * scale_factor
            volume = volume.astype(np.float32)

            label = sitk.ReadImage(label_path)
            # label = resize_image_itk(label, image_size, sitk.sitkNearestNeighbor)
            label = sitk.GetArrayFromImage(label)
            label = label[np.newaxis, ...]
            label = label_resize(label)
            label = label.squeeze(dim=0)
            label_value = np.unique(label).tolist()
            assert len(label_value) == len(label_map), 'label number error'
            for value in label_value:
                assert label_map.get(value, None) is not None, 'label_map error'
                label[label == value] = label_map.get(value, -1)
            print(np.unique(label))
            label = label.astype(np.uint8)

            image_group = file.create_group(dir_name)
            image_group.create_dataset(name='volume', data=volume)
            image_group.create_dataset(name='label', data=label)
    file.attrs['label_map'] = [[origin_label, target_label] for origin_label, target_label in label_map.items()]
    file.attrs['image_size'] = image_size
    file.attrs['dataset_size'] = len(file.keys())
    file.attrs['region_number'] = 6
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':

    write_LPBA40()
    hdf5_path = '../../datasets/hdf5/LPBA40.h5'
    output_dir = r'G:\biomdeical\registration\data\datasets'
    extract_hdf5(hdf5_path, output_dir)
