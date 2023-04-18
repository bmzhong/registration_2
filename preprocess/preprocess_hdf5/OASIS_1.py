import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *


def write_OASIS_1():
    output_path = '../../datasets/hdf5/OASIS_1.h5'
    if os.path.exists(output_path):
        os.remove(output_path)
    file = h5py.File(output_path, 'w')
    root_path = r'G:\biomdeical\registration\public_data\OASIS\neurite-oasis.v1.0'
    image_size = [160, 192, 224]
    scale_factor = 255.
    region_number = 4
    temp = []
    for dir_name in os.listdir(root_path):
        img_dir = os.path.join(root_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = 'norm.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = 'seg4.nii.gz'
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
            label.astype(np.uint8)

            print(np.unique(label))
            temp.append(len(np.unique(label)))

            image_group = file.create_group(dir_name)
            image_group.create_dataset(name='volume', data=volume)
            image_group.create_dataset(name='label', data=label)
    temp = np.array(temp)
    print(np.sum(temp == 5))
    print(len(temp))
    file.attrs['label_map'] = [[i, i] for i in range(0, region_number + 1)]
    file.attrs['image_size'] = image_size
    file.attrs['dataset_size'] = len(file.keys())
    file.attrs['region_number'] = region_number
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':
    write_OASIS_1()
    hdf5_path = '../../datasets/hdf5/OASIS_1.h5'
    output_dir = r'G:\biomdeical\registration\data\datasets'
    extract_hdf5(hdf5_path, output_dir)