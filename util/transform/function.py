import warnings
from collections.abc import Sequence

import monai
import torch
import numpy as np
import torchvision.transforms
from monai.transforms import Resample, create_grid
from monai.utils import ensure_tuple, issequenceiterable, fall_back_tuple


class BaseTransform(object):
    def transform(self, img, img_type=0):
        return img

    def __call__(self, img, img_type=0):
        if isinstance(img, Sequence):
            """
            img: [img, seg, img, seg, ...]
            img_type=0: image
            img_type=1: seg label
            """
            return [self.transform(img, i % 2) for i, img in enumerate(img)]
        return self.transform(img, img_type)


class Rotate90(BaseTransform):
    def __init__(self, axis=(0, 1)):
        """
        axis: Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.axis = axis
        self.rotate90 = monai.transforms.Rotate90(k=1, spatial_axes=self.axis)

    def transform(self, img, img_type=0):
        """
        img: C, D, H, W  or  C, H, W
        """
        return self.rotate90(img)


class RandRotate(BaseTransform):
    def __init__(self, angle=(0.5, 0.5, 0.5), prob=0.5):
        if not isinstance(angle, Sequence):
            angle = (angle,) * 3
        sample_angle = [np.random.uniform(-i, i) for i in angle]
        self.is_transform = True if np.random.uniform(0, 1.0) <= prob else False
        self.rotator_bilinear = monai.transforms.Rotate(
            angle=sample_angle,
            keep_size=True,
            mode='bilinear',
            padding_mode='zeros'
        )
        self.rotator_nearest = monai.transforms.Rotate(
            angle=sample_angle,
            keep_size=True,
            mode='nearest',
            padding_mode='zeros'
        )

    def transform(self, img, img_type=0):
        if not self.is_transform:
            return img

        if img_type == 0:
            return self.rotator_bilinear(img)

        elif img_type == 1:
            return self.rotator_nearest(img)
        else:
            raise Exception("img_type error in RandRotate")


class Flip(BaseTransform):
    def __init__(self, axis=None):
        self.flipper = monai.transforms.Flip(spatial_axis=axis)

    def transform(self, img, img_type=0):
        return self.flipper(img)


class RandFlip(BaseTransform):
    def __init__(self, prob=0.5, axis=None):
        self.is_transform = True if np.random.uniform(0, 1.0) <= prob else False
        self.flipper = monai.transforms.Flip(spatial_axis=axis)

    def transform(self, img, img_type=0):
        if self.is_transform:
            return self.flipper(img)
        return img


class RandAxisFlip(BaseTransform):
    def __init__(self, prob=0.5, ndim=3):
        self.is_transform = True if np.random.uniform(0, 1.0) <= prob else False
        self.axis = np.random.randint(0, ndim)
        self.flipper = monai.transforms.Flip(spatial_axis=self.axis)

    def transform(self, img, img_type=0):
        if self.is_transform:
            return self.flipper(img)
        return img


class Zoom(BaseTransform):
    def __init__(self, zoom=1.):
        self.zoom_bilinear = monai.transforms.Zoom(zoom=zoom, mode='bilinear')
        self.zoom_nearest = monai.transforms.Zoom(zoom=zoom, mode='nearest')

    def transform(self, img, img_type=0):
        if img_type == 0:
            return self.zoom_bilinear(img)
        elif img_type == 1:
            return self.zoom_nearest(img)
        else:
            raise Exception("img_type error in Zoom")


class RandZoom(BaseTransform):
    def __init__(self, prob=0.5, min_zoom=0.9, max_zoom=1.1):
        self.is_transform = True if np.random.uniform(0, 1.0) <= prob else False
        self.sample_zoom = np.random.uniform(min_zoom, max_zoom)
        self.zoom_bilinear = monai.transforms.Zoom(zoom=self.sample_zoom, mode='bilinear')
        self.zoom_nearest = monai.transforms.Zoom(zoom=self.sample_zoom, mode='nearest')

    def transform(self, img, img_type=0):
        if not self.is_transform:
            return img
        if img_type == 0:
            return self.zoom_bilinear(img)
        elif img_type == 1:
            return self.zoom_nearest(img)
        else:
            raise Exception("img_type error in RandZoom")


class Affine(BaseTransform):
    def __init__(self, rotate_params=None, shear_params=None, translate_params=None, scale_params=None):
        self.affine_bilinear = monai.transforms.Affine(rotate_params=rotate_params,
                                                       shear_params=shear_params,
                                                       translate_params=translate_params,
                                                       scale_params=scale_params,
                                                       mode='bilinear',
                                                       image_only=True)
        self.affine_nearest = monai.transforms.Affine(rotate_params=rotate_params,
                                                      shear_params=shear_params,
                                                      translate_params=translate_params,
                                                      scale_params=scale_params,
                                                      mode='nearest',
                                                      image_only=True)

    def transform(self, img, img_type=0):
        if img_type == 0:
            return self.affine_bilinear(img)
        elif img_type == 1:
            return self.affine_nearest(img)
        else:
            raise Exception("img_type error in Affine")


class RandAffineGrid():
    def __init__(
            self,
            rotate_range=None,
            shear_range=None,
            translate_range=None,
            scale_range=None,
            device=None,
    ):
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params = None
        self.shear_params = None
        self.translate_params = None
        self.scale_params = None

        self.device = device
        self.affine = torch.eye(4, dtype=torch.float64)
        self.randomize()
        self.affine_grid = monai.transforms.AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            device=self.device,
        )

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(np.random.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(np.random.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self):
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(self, spatial_size=None, grid=None):
        _grid, self.affine = self.affine_grid(spatial_size, grid)  # type: ignore
        return _grid


class RandAffine(BaseTransform):

    def __init__(
            self,
            prob: float = 0.1,
            rotate_range=None,
            shear_range=None,
            translate_range=None,
            scale_range=None,
            spatial_size=None
    ):
        self.is_transform = True if np.random.uniform(0, 1.0) <= prob else False
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
        )
        self.resampler = Resample()
        assert spatial_size is not None, 'spatial_size must be not None'
        self.spatial_size = spatial_size
        self.grid = self.rand_affine_grid(spatial_size=self.spatial_size)

    def transform(self, img, img_type=0):
        if not self.is_transform:
            return img
        if img_type == 0:
            return self.resampler(img=img, grid=self.grid, mode='bilinear', padding_mode='zeros')
        elif img_type == 1:
            return self.resampler(img=img, grid=self.grid, mode='nearest', padding_mode='zeros')
        else:
            raise Exception("img_type error in RandAffine")


class Resize(BaseTransform):
    def __init__(self, spatial_size):
        self.spatial_size = spatial_size
        self.resize_bilinear = monai.transforms.Resize(spatial_size=self.spatial_size, mode='bilinear')
        self.resize_nearest = monai.transforms.Resize(spatial_size=self.spatial_size, mode='nearest')

    def transform(self, img, img_type=0):
        if img_type == 0:
            return self.resize_bilinear(img)
        elif img_type == 1:
            return self.resize_nearest(img)
        else:
            raise Exception("img_type error in Resize")


class SpatialPad(BaseTransform):
    def __init__(self, spatial_size, value=0):
        self.spatial_size = spatial_size
        self.value = value
        self.pad_function = monai.transforms.SpatialPad(spatial_size=self.spatial_size, value=self.value)

    def transform(self, img, img_type=0):
        return self.pad_function(img)


class BorderPad(BaseTransform):
    def __init__(self, spatial_border, value=0):
        self.spatial_border = spatial_border
        self.value = value


# monai.transforms.RandRotateD
if __name__ == '__main__':
    import math

    a_dict = {
        "a": torch.arange(0, 20).reshape((1, 4, 5)).contiguous(),
        "b": None
    }
    transforms = monai.transforms.SpatialPadD(keys=["a", "b"],spatial_size=(10,10))
    res = transforms(a_dict)
    print(res)
    # a = torch.arange(0, 20).reshape((1, 4, 5)).contiguous()

    # tranforms = torchvision.transforms.Compose([RandAffine(prob=1,)])
