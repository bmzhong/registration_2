import torch
from model.registration.voxelmorpher.VoxelMorpher import VoxelMorpher


def get_reg_model(model_config, checkpoint=None):
    if model_config['type'] == 'VoxelMorpher':
        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 8, 8]
        reg_net = VoxelMorpher(3, nf_enc, nf_dec)

    else:
        raise Exception(f"There are no {model_config['type']}, please check.")

    if checkpoint is not None:
        reg_net.load_state_dict(torch.load(checkpoint)["model"])

    return reg_net
