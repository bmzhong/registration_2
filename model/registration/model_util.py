import monai
import torch
from model.registration.voxelmorpher.VoxelMorpher import VoxelMorpher
from model.registration.voxelmorpher.VoxelMorpher_2 import VoxelMorpher_2
from model.registration.voxelmorpher.VoxelMorpher_3 import VoxelMorpher_3


def get_reg_model(model_config, checkpoint=None):
    if model_config['type'] == 'VoxelMorpher':
        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 8, 8]
        reg_net = VoxelMorpher(3, nf_enc, nf_dec)
    elif model_config['type'] == 'VoxelMorpher_2':
        reg_net = VoxelMorpher_2()
    elif model_config['type'] == 'VoxelMorpher_3':
        reg_net = VoxelMorpher_3()
    else:
        raise Exception(f"There are no {model_config['type']}, please check.")

    if checkpoint is not None:
        reg_net.load_state_dict(torch.load(checkpoint)["model"])

    return reg_net
