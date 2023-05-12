import monai
import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *


def write_OASIS_1(image_size, scale_factor, source_path, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    file = h5py.File(output_path, 'w')

    region_number = 4
    volume_resize = monai.transforms.Resize(spatial_size=image_size, mode='trilinear', align_corners=False)
    label_resize = monai.transforms.Resize(spatial_size=image_size, mode='nearest', align_corners=None)
    temp = []
    for dir_name in tqdm(os.listdir(source_path)):
        img_dir = os.path.join(source_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = 'norm.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = 'seg4.nii.gz'
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

            label.astype(np.uint8)

            # print(np.unique(label))
            temp.append(len(np.unique(label)))

            image_group = file.create_group(dir_name)
            image_group.create_dataset(name='volume', data=volume)
            image_group.create_dataset(name='label', data=label)
    temp = np.array(temp)
    print(np.sum(temp == 5))
    print(len(temp))
    # file.attrs['label_map'] = [[i, i] for i in range(0, region_number + 1)]
    file.attrs['image_size'] = image_size
    file.attrs['dataset_size'] = len(file.keys())
    file.attrs['region_number'] = region_number
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':
    image_size = [128, 128, 128]
    scale_factor = 1.
    output_path = '../../datasets/hdf5/OASIS.h5'
    source_path = r'G:\biomdeical\registration\public_data\OASIS\neurite-oasis.v1.0'
    write_OASIS_1(image_size, scale_factor, source_path, output_path)

    hdf5_path = '../../datasets/hdf5/OASIS.h5'
    output_dir = r'G:\biomdeical\registration\data\datasets'
    extract_hdf5(hdf5_path, output_dir)
