import numpy as np

# def partition_eight_patch(image: np.ndarray):
#     D, H, W = image.shape
#     assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, 'image shape error'
#     images_list = []
#
#     def _partition(_image, _axis):
#         _image_1, _image_2 = np.split(_image, indices_or_sections=2, axis=_axis)
#         if _axis == 2:
#             images_list.append(_image_1)
#             images_list.append(_image_2)
#         else:
#             _partition(_image_1, _axis + 1)
#             _partition(_image_2, _axis + 1)
#
#     _partition(image, _axis=0)
#     return images_list


# def partition_patch_soma_nuclei():
#     root_path = r'G:\biomdeical\registration\data\preprocess\merge_label\soma_nuclei'
#     partition_dir = r'G:\biomdeical\registration\data\preprocess\partition\soma_nuclei'
#     for dir_name in os.listdir(root_path):
#         image_dir = os.path.join(root_path, dir_name)
#         volume = sitk.ReadImage(os.path.join(image_dir, dir_name + '.nii.gz'))
#         volume = sitk.GetArrayFromImage(volume)
#         volume_list = partition_eight_patch(volume)
#         label = sitk.ReadImage(os.path.join(image_dir, dir_name + '_label.nii.gz'))
#         label = sitk.GetArrayFromImage(label)
#         label_list = partition_eight_patch(label)
#         for i in range(1, len(volume_list) + 1):
#             volume_i = volume_list[i]
#             label_i = label_list[i]

def partition_eight_patch(image: np.ndarray):
    D, H, W = image.shape
    assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, 'image shape error'
    images_list = []

    def _partition(_image, _axis):
        _image_1, _image_2 = np.split(_image, indices_or_sections=2, axis=_axis)
        if _axis == 2:
            images_list.append(_image_1)
            images_list.append(_image_2)
        else:
            _partition(_image_1, _axis + 1)
            _partition(_image_2, _axis + 1)

    _partition(image, _axis=0)
    return images_list
    # image_up, image_down = np.split(image, 2, axis=0)
    # image_up_left, image_up_right = np.split(image_up, 2, axis=1)
    # image_down_left, image_down_right = np.split(image_down, 2, axis=1)


if __name__ == '__main__':
    arr1 = np.arange(4 ** 3).reshape((4, 4, 4))
    print(arr1)
    image_list = partition_eight_patch(arr1)
    for image in image_list:
        print(image.shape)
        print(image)
        print('----------------------------')

