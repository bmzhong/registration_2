import torch

from model.segmentation.unet.MonaiUnet import MonAI_Unet


def get_seg_model(model_config, num_classes, checkpoint=None):
    if model_config['type'] == 'UNet':
        seg_net = MonAI_Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(8, 16, 16, 32, 32, 64, 64),
            strides=(1, 2, 1, 2, 1, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=0,
            act='PRELU',
            norm='batch',
            dropout=0.2,
            bias=True,
            adn_ordering='NDA'
        )
    else:
        raise Exception(f"There are no {model_config['type']}, please check.")

    if checkpoint is not None:
        seg_net.load_state_dict(torch.load(checkpoint)["model"])

    return seg_net
