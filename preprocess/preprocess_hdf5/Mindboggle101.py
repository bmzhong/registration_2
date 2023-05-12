import monai
import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *

Mindboggle_group_config = [
    ("Frontal lobe", [1002, 1010, 1012, 1014, 1018, 1019, 1020, 1026, 1027, 1028] +
     [2002, 2010, 2012, 2014, 2018, 2019, 2020, 2026, 2027, 2028]),
    ("Parietal lobe", [1003, 1008, 1022, 1024, 1029] +
     [2003, 2008, 2022, 2024, 2029]),
    ("Temporal lobe", [1006, 1009, 1015, 1016, 1030] +
     [2006, 2009, 2015, 2016, 2030]),
    ("Occipital lobe", [1005, 1025, 1021] +
     [2005, 2025, 2021]),
    ("Insular cortex", [1007, 1011, 1013] +
     [2007, 2011, 2013]),
    ("Cingulate gyrus", [1023] + [2023]),
    ("Central sulcus", [1017] + [2017]),
    ("Lateral fissure", [1031, 1034, 1035] + [2031, 2034, 2035])
]


# def group_label(label):
#     label_merged = np.zeros(label.shape, dtype=np.int32)
#     for i, (name, id_list) in enumerate(Mindboggle_group_config):
#         # region = np.logical_and(label >= start, True)
#         region = np.argwhere(label in id_list)
#         label_merged[region] = i + 1
#     return label_merged


# def get_Mindboggle101_label_map():
#     source_path = r'G:\biomdeical\registration\public_data\MindBoggle101\MindBoggle101_from_official_webset' \
#                 r'\Mindboggle101_individuals\Mindboggle101_volumes\merge\HLN-12-1\labels.DKT31.manual.MNI152.nii.gz'
#     label = sitk.ReadImage(source_path)
# 
#     label = sitk.GetArrayFromImage(label)
#     label_unique = list(np.unique(label))
#     label_unique.sort()
#     label_map = dict()
#     for i in range(len(label_unique)):
#         label_map[label_unique[i]] = i
#     print(label_map)
#     return label_map


def write_Mindboggle101(image_size, source_path, output_path, scale_factor):
    label_map = dict()
    for i, (name, id_list) in enumerate(Mindboggle_group_config):
        for id in id_list:
            label_map[id] = i + 1

    if os.path.exists(output_path):
        os.remove(output_path)
    file = h5py.File(output_path, 'w')

    volume_resize = monai.transforms.Resize(spatial_size=image_size, mode='trilinear', align_corners=False)
    label_resize = monai.transforms.Resize(spatial_size=image_size, mode='nearest', align_corners=None)

    for dir_name in tqdm(os.listdir(source_path)):
        img_dir = os.path.join(source_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = 't1weighted_brain.MNI152.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = 'labels.DKT31.manual.MNI152.nii.gz'
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
            # label = group_label(label)
            label = label.astype(np.int32)

            new_label = np.zeros(label.shape, label.dtype)
            for origin_label, target_label in label_map.items():
                new_label[label == origin_label] = target_label
            label = new_label

            label_value = np.unique(label).tolist()
            print(label_value)
            assert len(label_value) == 9
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
    file.attrs['region_number'] = len(Mindboggle_group_config)
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':
    source_path = r'G:\biomdeical\registration\public_data\MindBoggle101\MindBoggle101_from_official_webset' \
                  r'\Mindboggle101_individuals\Mindboggle101_volumes\merge'
    output_path = '../../datasets/hdf5/Mindboggle101.h5'
    image_size = [128, 128, 128]
    scale_factor = 1.
    write_Mindboggle101(image_size, source_path, output_path, scale_factor)
    hdf5_path = '../../datasets/hdf5/Mindboggle101.h5'
    output_dir = r'G:\biomdeical\registration\data\datasets'
    extract_hdf5(hdf5_path, output_dir)
