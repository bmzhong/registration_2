import torch
from torch import nn
from torch.nn.functional import interpolate
import numpy as np


def conv3d_with_leakyReLU(*args):
    return nn.Sequential(nn.Conv3d(*args),
                         nn.LeakyReLU())


class VoxelMorpher_2(nn.Module):
    def __init__(self):
        super(VoxelMorpher_2, self).__init__()
        self.input_channel = 2
        self.encoder_0 = conv3d_with_leakyReLU(self.input_channel, 16, 3, 1, 1)
        self.encoder_1 = conv3d_with_leakyReLU(16, 32, 3, 2, 1)
        self.encoder_2 = conv3d_with_leakyReLU(32, 32, 3, 2, 1)
        self.encoder_3 = conv3d_with_leakyReLU(32, 32, 3, 2, 1)
        self.decoder_0 = conv3d_with_leakyReLU(32, 32, 3, 1, 1)
        self.decoder_1 = conv3d_with_leakyReLU(64, 32, 3, 1, 1)
        self.decoder_2 = conv3d_with_leakyReLU(64, 32, 3, 1, 1)
        self.decoder_3 = conv3d_with_leakyReLU(48, 32, 3, 1, 1)
        self.decoder_4 = conv3d_with_leakyReLU(32, 32, 3, 1, 1)
        self.decoder_5 = conv3d_with_leakyReLU(32 + self.input_channel, 16, 3, 1, 1)
        self.decoder_6 = conv3d_with_leakyReLU(16, 16, 3, 1, 1)
        self.decoder_7 = conv3d_with_leakyReLU(16, 3, 3, 1, 1)

        self.act = nn.Tanh()

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        start = x
        conv_0 = self.encoder_0(x)
        conv_0_shape = np.array(conv_0.shape[2:])
        conv_1 = self.encoder_1(conv_0)
        conv_1_shape = np.array(conv_1.shape[2:])
        conv_2 = self.encoder_2(conv_1)
        conv_2_shape = np.array(conv_2.shape[2:])
        conv_3 = self.encoder_3(conv_2)

        x = self.decoder_0(conv_3)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((interpolate(x, scale_factor=(conv_2_shape / x_shape).tolist(), mode='trilinear'), conv_2), dim=1)
        x = self.decoder_1(x)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((interpolate(x, scale_factor=(conv_1_shape / x_shape).tolist(), mode='trilinear'), conv_1), dim=1)
        x = self.decoder_2(x)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((interpolate(x, scale_factor=(conv_0_shape / x_shape).tolist(), mode='trilinear'), conv_0), dim=1)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = torch.cat((x, start), dim=1)
        x = self.decoder_5(x)
        x = self.decoder_6(x)
        x = self.decoder_7(x)
        x = self.act(x)

        return x
