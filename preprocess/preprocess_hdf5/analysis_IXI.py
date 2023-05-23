import pickle
import glob
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from collections import Counter


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# source_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data\**\*.pkl"
# output_path = r"G:\biomdeical\registration\public_data\IXI\IXI_data\IXI_data_nii\subject"
# paths = glob.glob(source_path, recursive=True)
# labels = []
# for path in tqdm(paths):
#     image, label = pkload(path)
#     labels.append(list(np.unique(label)))
#     base_name = os.path.basename(path).split('.')[0]
#     image = sitk.GetImageFromArray(image)
#     label = sitk.GetImageFromArray(label)
#     _path = os.path.join(output_path, base_name)
#     os.makedirs(_path, exist_ok=True)
#     sitk.WriteImage(image, os.path.join(_path, "volume.nii.gz"))
#     sitk.WriteImage(label, os.path.join(_path, "label.nii.gz"))

# with open("./IXI_label.pkl", 'wb') as f:
#     pickle.dump(labels, f)

path = "./IXI_label.pkl"
labels = pkload(path)
print(labels)
atlas = labels[0]
print("atlas: ", len(atlas))
# result = []
# lens = []
# min_index = -1
# for i, label in enumerate(labels):
#     lens.append(len(label))
#     common = set(atlas) & set(label)
#     # print(len(common))
#     result.append(len(common))
#     if len(common) == 36:
#         min_index = i
# lens_cnt=Counter(lens)
# print(lens_cnt)
# cnt = Counter(result)
# result = np.array(result)
# print("max: ", result.max())
# print("min: ", result.min())
# print(cnt)
# print(min_index)
# print(labels[min_index])
# print(set(labels[min_index]) & set(atlas))
# print(len(labels[min_index]))
# print(labels)

# res_label = {0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 24.0, 26.0, 28.0,
#              31.0, 41.0, 42.0, 43.0, 46.0, 47.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 58.0, 60.0, 63.0, 77.0, 85.0}
#
# path = "./IXI_label.pkl"
# labels = pkload(path)
# result = []
# for i, label in enumerate(labels):
#     common = res_label & set(label)
#     result.append(len(common))
# result = np.array(result)
# print(result.max())
# print(result.min())