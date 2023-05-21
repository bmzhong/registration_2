import monai
import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *


def write_OASIS_1(image_size, scale_factor, source_path, output_path, region_number):
    print(image_size,region_number)
    if os.path.exists(output_path):
        os.remove(output_path)
    file = h5py.File(output_path, 'w')
    volume_resize = monai.transforms.Resize(spatial_size=image_size, mode='trilinear', align_corners=False)
    label_resize = monai.transforms.Resize(spatial_size=image_size, mode='nearest', align_corners=None)
    temp = []
    for dir_name in tqdm(os.listdir(source_path)):
        img_dir = os.path.join(source_path, dir_name)
        if os.path.isdir(img_dir):
            volume_name = 'aligned_norm.nii.gz'
            volume_path = os.path.join(img_dir, volume_name)
            label_name = 'aligned_seg' + str(region_number) + '.nii.gz'
            label_path = os.path.join(img_dir, label_name)
            volume = sitk.ReadImage(volume_path)
            volume = sitk.GetArrayFromImage(volume)
            volume = volume.T
            if volume.shape != list(image_size):
                volume = volume[np.newaxis, ...]
                volume = volume_resize(volume)
                volume = volume.squeeze(dim=0)

            volume = ((volume - volume.min()) / (volume.max() - volume.min())) * scale_factor

            volume = volume.astype(np.float32)

            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            label = label.T

            if label.shape != list(image_size):
                label = label[np.newaxis, ...]
                label = label_resize(label)
                label = label.squeeze(dim=0)
            # print(np.unique(label))
            temp.append(len(np.unique(label)))
            label.astype(np.uint8)
            image_group = file.create_group(dir_name)
            image_group.create_dataset(name='volume', data=volume)
            image_group.create_dataset(name='label', data=label)
    temp = np.array(temp)
    print(np.sum(temp == 36))
    print(len(temp))
    # file.attrs['label_map'] = [[i, i] for i in range(0, region_number + 1)]
    file.attrs['image_size'] = image_size
    file.attrs['dataset_size'] = len(file.keys())
    file.attrs['region_number'] = region_number
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':
    image_size = [160, 192, 224]
    scale_factor = 1.
    region_number = 4
    # output_path = '../../datasets/hdf5/35_192_OASIS.h5'
    output_path = '../../datasets/hdf5/'+str(region_number)+'_'+str(max(image_size))+'_OASIS.h5'
    print(output_path)
    source_path = r'G:\biomdeical\registration\public_data\OASIS\neurite-oasis.v1.0'
    write_OASIS_1(image_size, scale_factor, source_path, output_path, region_number)

    # hdf5_path = '../../datasets/hdf5/OASIS.h5'
    # output_dir = r'G:\biomdeical\registration\data\datasets'
    # extract_hdf5(hdf5_path, output_dir)
