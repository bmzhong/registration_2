import torch
from torch import nn
import monai
from monai.networks.blocks import PatchEmbed, TransformerBlock
import numpy as np
from model.mae.util import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed
from functools import partial


class MaskedAutoencoderViT3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, spatial_dims=3):
        super(MaskedAutoencoderViT3D, self).__init__()
        self.img_size = [img_size, ] * spatial_dims if isinstance(img_size, int) else img_size
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, norm_layer=None, spatial_dims=spatial_dims)
        self.num_patches = np.cumprod(np.array(self.img_size) // patch_size)[-1]
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=embed_dim, mlp_dim=int(mlp_ratio * embed_dim), num_heads=num_heads,
                              dropout_rate=0., qkv_bias=True) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(hidden_size=decoder_embed_dim, mlp_dim=int(mlp_ratio * decoder_embed_dim),
                             num_heads=decoder_num_heads, qkv_bias=True) for i in
            range(decoder_depth)]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** spatial_dims * in_chans,
                                      bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.spatial_dims = spatial_dims
        self.initialize_weights()

    def initialize_weights(self):
        print("initialize weights")
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], np.array(self.img_size) // self.patch_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], np.array(self.img_size) // self.patch_size,
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        c = imgs.shape[1]
        p = self.patch_size
        d = imgs.shape[2] // p
        h = imgs.shape[3] // p
        w = imgs.shape[4] // p
        x = imgs.reshape((imgs.shape[0], c, d, p, h, p, w, p))
        x = torch.einsum('ncdthpwq->ndhwtpqc', x)
        x = x.reshape((imgs.shape[0], d * h * w, p * p * p * c))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        d, h, w = np.array(self.img_size) // self.patch_size
        x = x.reshape((x.shape[0], d, h, w, p, p, p, self.in_chans))
        x = torch.einsum('ndhwtpqc->ncdthpwq', x)
        imgs = x.reshape((x.shape[0], self.in_chans, d * p, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.01):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, **kwargs)
    return model


if __name__ == "__main__":
    model = mae_vit_base_patch16_dec512d8b(in_chans=1, img_size=128)
    imgs = torch.empty((2, 1, 128, 128, 128))
    loss, pred, mask = model(imgs)
    patchify_imgs = model.patchify(imgs)
    print(patchify_imgs.shape)
    mask_image = patchify_imgs * mask.unsqueeze(dim=-1)
    mask_image = model.unpatchify(mask_image)
    print(mask_image.shape)
    # print(pred.shape)
    # pred1 = model.unpatchify(pred)
    # print(pred1.shape)

    # from torchsummary import summary
    #
    # summary(model, (1, 128, 128, 128), batch_size=2, device='cpu')
