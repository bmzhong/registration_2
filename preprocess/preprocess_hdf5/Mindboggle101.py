import monai
import numpy as np
from preprocess.preprocess_hdf5.hdf5_utils import *

# label 25
# GROUP_CONFIG = {
#     'parietal_lobe_1': ['postcentral', 'supramarginal', 'superior parietal', 'inferior parietal', ' precuneus'],
#     'frontal_lobe_2': ['superior frontal', 'middle frontal', 'inferior frontal', 'lateral orbitofrontal',
#                        'medial orbitofrontal', 'precentral', 'paracentral'],
#     'occipital_lobe_3': ['lingual', 'pericalcarine', 'cuneus', 'lateral occipital'],
#     'temporal_lobe_4': ['entorhinal', 'parahippocampal', 'fusiform', 'superior temporal', 'middle temporal',
#                         'inferior temporal', 'transverse temporal'],
#     'cingulate_lobe_5': ['cingulate', 'insula'],
# }
# label 31
GROUP_CONFIG = {
    'parietal_lobe_1': ['postcentral', 'supramarginal', 'superior parietal', 'inferior parietal', ' precuneus'],
    'frontal_lobe_2': ['caudal middle frontal', 'lateral orbitofrontal', 'medial orbitofrontal', 'paracentral',
                       'pars opercularis', 'pars orbitalis', 'pars triangularis', 'precentral',
                       'rostral middle frontal', 'superior frontal'],
    'occipital_lobe_3': ['lingual', 'pericalcarine', 'cuneus', 'lateral occipital'],
    'temporal_lobe_4': ['entorhinal', 'parahippocampal', 'fusiform', 'superior temporal', 'middle temporal',
                        'inferior temporal', 'transverse temporal'],
    'cingulate_lobe_5': ['caudal anterior cingulate', 'insula', 'isthmus cingulate', 'posterior cingulate',
                         'rostral anterior cingulate'],
}


def get_Mindboggle101_label_map():
    label_name = './mind101_label_31.txt'
    d = {}
    with open(label_name) as f:
        for line in f:
            if line != '\r\n':
                (value, key) = line.strip().split(',')
                d[key.strip().strip('"')] = int(value)
    name_to_new_id = dict()
    for key in GROUP_CONFIG:
        label_id = int(key.split('_')[-1])
        for structure in GROUP_CONFIG[key]:
            name_to_new_id['left ' + structure.strip()] = label_id
            name_to_new_id['right ' + structure.strip()] = label_id
    label_map = dict()
    for name in name_to_new_id.keys():
        label_map[d[name]] = name_to_new_id[name]
    return label_map


def write_Mindboggle101(image_size, source_path, output_path, scale_factor):
    label_map = get_Mindboggle101_label_map()

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
    file.attrs['region_number'] = len(np.unique(list(label_map.values())))
    file.attrs['normalize'] = [0, scale_factor]
    file.close()


if __name__ == '__main__':
    a=get_Mindboggle101_label_map()
    print(len(a))
    # source_path = r'G:\biomdeical\registration\public_data\MindBoggle101\MindBoggle101_from_official_webset' \
    #               r'\Mindboggle101_individuals\Mindboggle101_volumes\merge'
    # output_path = '../../datasets/hdf5/5_192_Mindboggle101.h5'
    # image_size = [160, 192, 160]
    # scale_factor = 1.
    # write_Mindboggle101(image_size, source_path, output_path, scale_factor)
#     hdf5_path = '../../datasets/hdf5/Mindboggle101.h5'
#     output_dir = r'G:\biomdeical\registration\data\datasets'
#     extract_hdf5(hdf5_path, output_dir)
