import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt

path = r"G:\biomdeical\registration\public_data\LPBA40\LPBA40_Subjects_Delineation_Space_MRI_and_label_files\LPBA40subjects.delineation_space\LPBA40\delineation_space\S06"
volume_path = os.path.join(path, "S06.delineation.skullstripped.img.gz")
label_path = os.path.join(path, "S06.delineation.structure.label.img.gz")
volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
D, H, W = volume.shape
slice1 = np.concatenate((volume[D // 2, :, :], label[D // 2, :, :]), axis=1)
slice2 = np.concatenate((volume[:, H // 2, :], label[:, H // 2, :]), axis=1)
slice3 = np.concatenate((volume[:, :, W // 2], label[:, :, W // 2]), axis=1)
# a_slice = np.concatenate((slice1, slice2, slice3), axis=1)

plt.figure(figsize=(30, 30))

plt.imshow(slice1)
plt.axis('off')  # 去掉坐标轴
plt.show()
plt.imshow(slice2)
plt.axis('off')  # 去掉坐标轴
plt.show()
plt.imshow(slice3)
plt.axis('off')  # 去掉坐标轴
plt.show()
