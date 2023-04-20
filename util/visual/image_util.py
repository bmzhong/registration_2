import os

import SimpleITK as sitk

from util.visual.visual_image import preview_image
import matplotlib.pyplot as plt
import os

from util.visual.visual_registration import preview_3D_deformation, preview_3D_vector_field


def write_image(outdir, name, img, type):
    os.makedirs(outdir, exist_ok=True)
    if len(type) > 0:
        name = name + '_' + type
    data_type = 'float'
    if type == 'label':
        data_type = 'uint8'
    img = img.cpu().numpy().astype(data_type)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, os.path.join(outdir, name + '.nii.gz'))


def save_image_figure(outdir, name, image, cmap='gray'):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_image(image, cmap=cmap)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))


def save_deformation_figure(outdir, name, dvf, grid_spacing, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_3D_deformation(dvf, grid_spacing, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))


def save_dvf_figure(outdir, name, dvf):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_3D_vector_field(dvf)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))


def save_det_figure(outdir, name, det, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    figure = preview_image(det, **kwargs)
    plt.suptitle(name, fontsize=25)
    figure.tight_layout()
    plt.savefig(os.path.join(outdir, name + '.jpg'))
