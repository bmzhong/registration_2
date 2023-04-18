import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *


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
    # label_map = get_Mindboggle101_label_map()
    label_map = {

    }
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
