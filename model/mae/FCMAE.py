import numpy as np
import torch
from torch import nn
from torch.nn.init import trunc_normal_

from model.mae.UNet_Module import UNet_Encoder, UNet_Decoder


class FCMAE(nn.Module):
    def __init__(self,
                 enc_nf,
                 dec_nf,
                 in_chans,
                 patch_size,
                 decoder_embed_dim,
                 mask_ratio=0.6,
                 ndims=3,
                 norm_pix_loss=False):
        super(FCMAE, self).__init__()
        self.patch_size = [patch_size] * ndims if isinstance(patch_size, int) else patch_size
        self.ndims = ndims
        scale = int(np.log2(self.patch_size[0]))
        self.encoder = UNet_Encoder(enc_nf, in_chans, num_downsample=scale, ndims=self.ndims)
        Conv = getattr(nn, 'Conv%dd' % self.ndims)
        self.proj = Conv(
            in_channels=enc_nf[-1],
            out_channels=decoder_embed_dim,
            kernel_size=1)
        self.decoder = UNet_Decoder(dec_nf, decoder_embed_dim, num_upsample=scale, ndims=self.ndims)
        self.mask_token = nn.Parameter(torch.zeros((1, decoder_embed_dim) + (1,) * self.ndims))
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if self.ndims == 2:
            c = imgs.shape[1]
            p_h, p_w = self.patch_size
            h = imgs.shape[2] // p_h
            w = imgs.shape[3] // p_w
            x = imgs.reshape((imgs.shape[0], c, h, p_h, w, p_w))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape((imgs.shape[0], h * w, p_h * p_w * c))
        elif self.ndims == 3:
            c = imgs.shape[1]
            p_d, p_h, p_w = self.patch_size
            d = imgs.shape[2] // p_d
            h = imgs.shape[3] // p_h
            w = imgs.shape[4] // p_w
            x = imgs.reshape((imgs.shape[0], c, d, p_d, h, p_h, w, p_w))
            x = torch.einsum('ncdthpwq->ndhwtpqc', x)
            x = x.reshape((imgs.shape[0], d * h * w, p_d * p_h * p_w * c))
        else:
            raise Exception
        return x

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        mask_shape = np.array(x.shape[2:]) // np.array(self.patch_size)
        L = np.cumprod(mask_shape)[-1]

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, x, mask):

        N = x.shape[0]
        mask_shape = np.array(x.shape[2:]) // np.array(self.patch_size)
        mask = mask.reshape(N, *mask_shape).contiguous()
        mask = mask.unsqueeze(1)
        for i in range(len(x.shape[2:])):
            mask = mask.repeat_interleave(x.shape[2 + i] // mask.shape[2 + i], axis=2 + i)
        return mask

    def forward_encoder(self, x, mask):

        mask = self.upsample_mask(x, mask)
        mask = mask.type_as(x)
        x = x * (1. - mask)
        x = self.encoder(x)
        return x

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        mask_token = self.mask_token.repeat((x.shape[0], 1, *x.shape[2:]))
        mask = mask.reshape(-1, *x.shape[2:])
        mask = mask.unsqueeze(dim=1).type_as(x)
        x = x * (1. - mask) + mask_token * mask
        x = self.decoder(x)
        return x

    def forward_loss(self, imgs, pred, mask):
        if pred.shape == imgs.shape:
            pred = self.patchify(pred)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        mask = self.gen_random_mask(imgs, self.mask_ratio)
        x = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


if __name__ == "__main__":
    enc_nf = [16, 32, 64, 128, 256, 128, 64]
    dec_nf = [32, 32, 32, 32, 16, 8, 1]
    x = torch.empty((2, 1, 160, 192, 160))
    model = FCMAE(enc_nf=enc_nf, dec_nf=dec_nf, in_chans=1, patch_size=16, decoder_embed_dim=32)
    loss, pred, mask = model(x)
    print(mask)
    print(mask.shape)
    import matplotlib.pyplot as plt

    # plt.subplot(231)
    # plt.imshow(output[0, 0, 64, :, :], cmap='gray')
    # plt.subplot(232)
    # plt.imshow(output[0, 0, :, 64, :], cmap='gray')
    # plt.subplot(233)
    # plt.imshow(output[0, 0, :, :, 64], cmap='gray')
    # plt.subplot(234)
    # plt.imshow(output[1, 0, 64, :, :], cmap='gray')
    # plt.subplot(235)
    # plt.imshow(output[1, 0, :, 64, :], cmap='gray')
    # plt.subplot(236)
    # plt.imshow(output[1, 0, :, :, 64], cmap='gray')
    # plt.show()
