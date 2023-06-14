import torch
from torch import nn
import numpy as np
from torchsummary import summary


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class UNet_Encoder(nn.Module):
    def __init__(self,
                 enc_nf,
                 infeats,
                 num_downsample,
                 ndims=3,
                 max_pool=2,
                 nb_conv_per_level=1,
                 ):
        super(UNet_Encoder, self).__init__()
        final_convs = enc_nf[num_downsample:]
        enc_nf = enc_nf[:num_downsample]
        nb_dec_convs = len(enc_nf)
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

    def forward(self, x):

        # encoder forward pass
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x = self.pooling[level](x)
        for conv in self.remaining:
            x = conv(x)
        return x


class UNet_Decoder(nn.Module):
    def __init__(self,
                 dec_nf=None,
                 infeats=None,
                 num_upsample=None,
                 ndims=3,
                 up_scale=2,
                 nb_conv_per_level=1,
                 ):
        super(UNet_Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        nb_dec_convs = num_upsample
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        if isinstance(up_scale, int):
            up_scale = [up_scale] * self.nb_levels
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in up_scale]
        prev_nf = infeats
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)

        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

    def forward(self, x):
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
                x = self.upsampling[level](x)
        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
        return x


if __name__ == "__main__":
    # encoder = UNet_Encoder(enc_nf=[16, 32, 32, 32], infeats=1)
    # print(encoder.state_dict().keys())
    # summary(encoder, (1, 128, 128, 128), batch_size=1, device="cpu")
    decoder = UNet_Decoder(dec_nf=[32, 32, 32, 32], infeats=1)
    print(decoder)
    # summary(decoder,(1, 128, 128, 128), batch_size=1, device="cpu")
