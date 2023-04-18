import os

import SimpleITK as sitk


def write_image(outdir, name, img, type):
    outdir = os.path.join(outdir, name)
    os.makedirs(outdir, exist_ok=True)
    if len(type) > 0:
        name = name + '_' + type
    data_type = 'float'
    if type == 'label':
        data_type = 'uint8'
    img = img[0].cpu().numpy().astype(data_type)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, os.path.join(outdir, name + '.nii.gz'))
