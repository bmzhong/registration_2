import torch
from torch import nn
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # use half of dimensions to encode grid_h
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, E_D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, E_D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim - embed_dim // 3 - embed_dim // 3, grid[2])  # (D*H*W, E_D/3)
    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    if embed_dim % 2:
        remaining = np.zeros((emb.shape[0], 1), dtype=emb.dtype)
        emb = np.concatenate([emb, remaining], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':
    pos_emb = get_3d_sincos_pos_embed(64, (128, 128, 128), True)
