import monai
import torch
import torchvision

prob = 1
transform_seg1 = torchvision.transforms.Compose([
    monai.transforms.RandAxisFlipD(keys=['volume', 'label'], prob=prob, allow_missing_keys=True),
    monai.transforms.RandRotateD(keys=['volume', 'label'], prob=prob, allow_missing_keys=True,
                                 range_x=0.5, range_y=0.5, range_z=0.5, mode=['bilinear', 'nearest'],
                                 padding_mode=['zeros', 'zeros']),
    monai.transforms.RandAffineD(keys=['volume', 'label'], prob=prob, allow_missing_keys=True,
                                 rotate_range=(-0.1, 0.1), translate_range=(-10, 10), scale_range=(0.9, 1.1),
                                 mode=['bilinear', 'nearest'], padding_mode=['zeros', 'zeros']),
    monai.transforms.RandZoomD(keys=['volume', 'label'], prob=prob, allow_missing_keys=True,
                               min_zoom=0.9, max_zoom=1.1, mode=['bilinear', 'nearest'],
                               padding_mode=['constant', 'constant']),
    # monai.transforms.RandGaussianSmoothD()
])

if __name__ == '__main__':
    import SimpleITK as sitk

    volume_path = r'../../datasets/temp_data/OASIS_OAS1_0001_MR1/volume.nii.gz'
    label_path = r'../../datasets/temp_data/OASIS_OAS1_0001_MR1/label.nii.gz'
    volume = sitk.ReadImage(volume_path)
    volume = sitk.GetArrayFromImage(volume)
    label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label)
    volume = torch.from_numpy(volume)
    label = torch.from_numpy(label)
    img = {'volume': volume, 'label': label}
    img = transform_seg1(img)
    volume = sitk.GetImageFromArray(img['volume'])
    label = sitk.GetImageFromArray(img['label'])
    volume = sitk.WriteImage(volume, './temp_volume.nii.gz')
    label = sitk.WriteImage(label, './temp_label.nii.gz')
