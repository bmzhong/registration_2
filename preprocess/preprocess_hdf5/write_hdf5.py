# from torchvision.transforms import functional as F
import os

import numpy as np

from preprocess.preprocess_hdf5.hdf5_utils import *





def test():
    # root_path = r'G:\biomdeical\registration\public_data\LPBA40\LPBA40_Subjects_Delineation_Space_MRI_and_label_files\LPBA40subjects.delineation_space\LPBA40\delineation_space\S01\S01.delineation.skullstripped.img.gz'
    root_path = '../../datasets/temp_data/S01/S01.delineation.structure.label.img.gz'
    image_size = [160, 192, 160]
    image = sitk.ReadImage(root_path)
    image = resize_image_itk(image, image_size, sitk.sitkNearestNeighbor)
    sitk.WriteImage(image, '../../datasets/temp_data/resample_S01_label.nii.gz')


def get_Mindboggle101_label_map():
    root_path = r'G:\biomdeical\registration\public_data\MindBoggle101\MindBoggle101_from_official_webset' \
                r'\Mindboggle101_individuals\Mindboggle101_volumes\merge\HLN-12-1\labels.DKT31.manual.MNI152.nii.gz'
    label = sitk.ReadImage(root_path)

    label = sitk.GetArrayFromImage(label)
    label_unique = list(np.unique(label))
    label_unique.sort()
    label_map = dict()
    for i in range(len(label_unique)):
        label_map[label_unique[i]] = i
    print(label_map)
    return label_map


def write_Mindboggle101():
    output_path = '../../datasets/hdf5/Mindboggle101.h5'
    if os.path.exists(output_path):
        os.remove(output_path)
    file = h5py.File(output_path, 'w')
    root_path = r'G:\biomdeical\registration\public_data\MindBoggle101\MindBoggle101_from_official_webset' \
                r'\Mindboggle101_individuals\Mindboggle101_volumes\merge'
    image_size = [160, 192, 160]
    scale_factor = 255.
    label_map = get_Mindboggle101_label_map()
    for dir_name in os.listdir(root_path):
        img_dir = os.path.join(root_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = 't1weighted_brain.MNI152.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = 'labels.DKT31.manual.MNI152.nii.gz'
            label_path = os.path.join(img_dir, label_name)

            volume = sitk.ReadImage(volume_path)
            volume = resize_image_itk(volume, image_size, sitk.sitkLinear)
            volume = sitk.GetArrayFromImage(volume)
            volume = volume.astype(np.float64)
            volume = ((volume - volume.min()) / (volume.max() - volume.min())) * scale_factor

            volume = volume.astype(np.float32)

            label = sitk.ReadImage(label_path)
            label = resize_image_itk(label, image_size, sitk.sitkNearestNeighbor)
            label = sitk.GetArrayFromImage(label)
            print(np.unique(label))
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
    # test()
    # write_LPBA40()
    # write_OASIS_1()
    # write_Mindboggle101()
    # hdf5_path = '../../datasets/hdf5/Mindboggle101.h5'
    # output_dir = r'G:\biomdeical\registration\data\datasets'
    # extract_hdf5(hdf5_path, output_dir)

    # hdf5_path = '../../datasets/hdf5/OASIS_1.h5'
    # output_dir = r'G:\biomdeical\registration\data\datasets'
    # extract_hdf5(hdf5_path, output_dir)
    # a = {i: 6 for i in range(161, 167)}
    # print(a)
    # pass
    cortex_numbers_names =  [
        [1002,    "left caudal anterior cingulate"],
        [1003,    "left caudal middle frontal"],
        [1005,    "left cuneus"],
        [1006,    "left entorhinal"],
        [1007,    "left fusiform"],
        [1008,    "left inferior parietal"],
        [1009,    "left inferior temporal"],
        [1010,    "left isthmus cingulate"],
        [1011,    "left lateral occipital"],
        [1012,    "left lateral orbitofrontal"],
        [1013,    "left lingual"],
        [1014,    "left medial orbitofrontal"],
        [1015,    "left middle temporal"],
        [1016,    "left parahippocampal"],
        [1017,    "left paracentral"],
        [1018,    "left pars opercularis"],
        [1019,    "left pars orbitalis"],
        [1020,    "left pars triangularis"],
        [1021,    "left pericalcarine"],
        [1022,    "left postcentral"],
        [1023,    "left posterior cingulate"],
        [1024,    "left precentral"],
        [1025,    "left precuneus"],
        [1026,    "left rostral anterior cingulate"],
        [1027,    "left rostral middle frontal"],
        [1028,    "left superior frontal"],
        [1029,    "left superior parietal"],
        [1030,    "left superior temporal"],
        [1031,    "left supramarginal"],
        [1034,    "left transverse temporal"],
        [1035,    "left insula"],
        [2002,    "right caudal anterior cingulate"],
        [2003,    "right caudal middle frontal"],
        [2005,    "right cuneus"],
        [2006,    "right entorhinal"],
        [2007,    "right fusiform"],
        [2008,    "right inferior parietal"],
        [2009,    "right inferior temporal"],
        [2010,    "right isthmus cingulate"],
        [2011,    "right lateral occipital"],
        [2012,    "right lateral orbitofrontal"],
        [2013,    "right lingual"],
        [2014,    "right medial orbitofrontal"],
        [2015,    "right middle temporal"],
        [2016,    "right parahippocampal"],
        [2017,    "right paracentral"],
        [2018,    "right pars opercularis"],
        [2019,    "right pars orbitalis"],
        [2020,    "right pars triangularis"],
        [2021,    "right pericalcarine"],
        [2022,    "right postcentral"],
        [2023,    "right posterior cingulate"],
        [2024,    "right precentral"],
        [2025,    "right precuneus"],
        [2026,    "right rostral anterior cingulate"],
        [2027,    "right rostral middle frontal"],
        [2028,    "right superior frontal"],
        [2029,    "right superior parietal"],
        [2030,    "right superior temporal"],
        [2031,    "right supramarginal"],
        [2034,    "right transverse temporal"],
        [2035,    "right insula"]]
    print([i[1] for i in cortex_numbers_names])