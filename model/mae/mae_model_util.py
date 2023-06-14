import torch

from model.mae.FCMAE import FCMAE
from model.mae.MaskedAutoencoderViT import mae_vit_base_patch16_dec512d8b


def get_mae_model(model_config, image_size, checkpoint=None):
    if model_config['type'] == 'UNet':
        enc_nf = [16, 32, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16, 1]
        model = FCMAE(enc_nf=enc_nf, dec_nf=dec_nf, in_chans=1, patch_size=16, decoder_embed_dim=32)
    elif model_config['type'] == 'vit_base':
        model = mae_vit_base_patch16_dec512d8b(in_chans=1, img_size=image_size)
    else:
        raise Exception(f"There are no {model_config['type']}, please check.")

    if checkpoint is not None:
        print(f"load weights from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint)["model"])

    return model
