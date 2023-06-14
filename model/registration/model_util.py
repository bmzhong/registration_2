import torch
from model.registration.voxelmorpher.VoxelMorpher import VxmDense
from model.registration.voxelmorpher.VoxelMorpher_0 import VoxelMorpher
from model.registration.xmorpher.XMorpher import XMorpherReg
from model.registration.voxelmorpher import VoxelMorph


def get_reg_model(model_config, checkpoint=None, image_size=(128, 128, 128)):
    if  model_config['type'] == 'VoxelMorph':
        reg_net = create_VoxelMorph(image_size)

    elif model_config['type'] == 'VoxelMorph_diff':
        reg_net = create_VoxelMorph_diff(image_size)

    elif model_config['type'] == 'VoxelMorph_prob':
        reg_net = create_VoxelMorph_prob(image_size)

    elif model_config['type'] == 'VoxelMorph_diff_prob':
        reg_net = create_VoxelMorph_diff_prob(image_size)

    elif model_config['type'] == 'XMorpherReg':
        reg_net = XMorpherReg(n_channels=1, patch_size=(4, 4, 4), scale=1)
    else:
        raise Exception(f"There are no {model_config['type']}, please check.")

    if checkpoint is not None:
        print(f"load weights from {checkpoint}")
        reg_net.load_state_dict(torch.load(checkpoint)["model"])

    return reg_net


def create_VoxelMorph(image_size=(128, 128, 128)):
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    # dec_nf = [32, 32, 32, 32, 8, 8]
    inshape = image_size
    int_steps = 0
    use_probs = False
    src_feats = 1
    trg_feats = 1
    return VoxelMorph.VxmDense(inshape=inshape,
                               nb_unet_features=[enc_nf, dec_nf],
                               int_steps=int_steps,
                               use_probs=use_probs,
                               src_feats=src_feats,
                               trg_feats=trg_feats
                               )


def create_VoxelMorph_diff(image_size=(128, 128, 128)):
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    # dec_nf = [32, 32, 32, 32, 8, 8]
    inshape = image_size
    int_steps = 7
    use_probs = False
    src_feats = 1
    trg_feats = 1
    return VoxelMorph.VxmDense(inshape=inshape,
                               nb_unet_features=[enc_nf, dec_nf],
                               int_steps=int_steps,
                               use_probs=use_probs,
                               src_feats=src_feats,
                               trg_feats=trg_feats
                               )


def create_VoxelMorph_prob(image_size=(128, 128, 128)):
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    # dec_nf = [32, 32, 32, 32, 8, 8]
    inshape = image_size
    int_steps = 0
    use_probs = True
    src_feats = 1
    trg_feats = 1
    return VoxelMorph.VxmDense(inshape=inshape,
                               nb_unet_features=[enc_nf, dec_nf],
                               int_steps=int_steps,
                               use_probs=use_probs,
                               src_feats=src_feats,
                               trg_feats=trg_feats
                               )


def create_VoxelMorph_diff_prob(image_size=(128, 128, 128)):
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    # dec_nf = [32, 32, 32, 32, 8, 8]
    inshape = image_size
    int_steps = 7
    use_probs = True
    src_feats = 1
    trg_feats = 1
    return VoxelMorph.VxmDense(inshape=inshape,
                               nb_unet_features=[enc_nf, dec_nf],
                               int_steps=int_steps,
                               use_probs=use_probs,
                               src_feats=src_feats,
                               trg_feats=trg_feats
                               )


def create_vxm_model(image_size=(128, 128, 128)):
    vxm_config = dict()
    vxm_config['inshape'] = image_size
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 8, 8]
    vxm_config['enc_nf'] = enc_nf
    vxm_config['dec_nf'] = dec_nf
    vxm_config['bn'] = None
    vxm_config['int_steps'] = 0
    vxm_config['int_downsize'] = 2
    vxm_config['bidir'] = False
    vxm_config['use_probs'] = False
    return VxmDense(**vxm_config)


def create_vxm_diff_model(image_size=(128, 128, 128)):
    vxm_config = dict()
    vxm_config['inshape'] = image_size
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 8, 8]
    vxm_config['enc_nf'] = enc_nf
    vxm_config['dec_nf'] = dec_nf
    vxm_config['bn'] = None
    vxm_config['int_steps'] = 7
    vxm_config['int_downsize'] = 2
    vxm_config['bidir'] = False
    vxm_config['use_probs'] = False

    return VxmDense(**vxm_config)


def create_vxm_prob_model(image_size=(128, 128, 128)):
    vxm_config = dict()
    vxm_config['inshape'] = image_size
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 8, 8]
    vxm_config['enc_nf'] = enc_nf
    vxm_config['dec_nf'] = dec_nf
    vxm_config['bn'] = None
    vxm_config['int_steps'] = 0
    vxm_config['int_downsize'] = 2
    vxm_config['bidir'] = False
    vxm_config['use_probs'] = True
    return VxmDense(**vxm_config)


def create_vxm_diff_prob_model(image_size=(128, 128, 128)):
    vxm_config = dict()
    vxm_config['inshape'] = image_size
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 8, 8]
    vxm_config['enc_nf'] = enc_nf
    vxm_config['dec_nf'] = dec_nf
    vxm_config['bn'] = None
    vxm_config['int_steps'] = 7
    vxm_config['int_downsize'] = 2
    vxm_config['bidir'] = False
    vxm_config['use_probs'] = True
    return VxmDense(**vxm_config)
