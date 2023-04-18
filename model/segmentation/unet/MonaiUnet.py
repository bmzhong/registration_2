import monai
import torch
from torch import nn


def MonAI_Unet(spatial_dims,
               in_channels,
               out_channels,
               channels,
               strides,
               kernel_size=3,
               up_kernel_size=3,
               num_res_units=0,
               act='PRELU',
               norm='INSTANCE',
               dropout=0.0,
               bias=True,
               adn_ordering='NDA',
               ):
    return monai.networks.nets.UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        adn_ordering=adn_ordering,
    )


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    seg_net = MonAI_Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=10,
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
    ).to(device)

    input = torch.randn((1, 1, 160, 192, 224)).to(device)
    output = seg_net(input)
    print(isinstance(seg_net, nn.Module))
    print(output.shape)
