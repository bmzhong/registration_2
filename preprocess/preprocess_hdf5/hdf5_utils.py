
import h5py
import SimpleITK as sitk
import os



def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    """
    https://blog.csdn.net/jancis/article/details/106265602
    """
    resampler = sitk.ResampleImageFilter()
    # originSize = itkimage.GetSize()  # 原来的体素块尺寸
    # originSpacing = itkimage.GetSpacing()
    # newSize = np.array(newSize, float)
    # factor = originSize / newSize
    # newSpacing = originSpacing * factor
    # newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize)
    # resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def extract_hdf5(hdf5_path, output_dir):
    datasets = h5py.File(hdf5_path, 'r')
    output_dir = os.path.join(output_dir, os.path.basename(hdf5_path).split('.')[0])
    for key1 in datasets.keys():
        for key2 in datasets[key1].keys():
            array = datasets[key1][key2][:]
            image = sitk.GetImageFromArray(array)
            output_path = os.path.join(output_dir, key1)
            os.makedirs(output_path,exist_ok=True)
            sitk.WriteImage(image, os.path.join(output_path, key2 + '.nii.gz'))
    datasets.close()
