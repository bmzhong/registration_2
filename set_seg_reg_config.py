import argparse
import os

import yaml


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--train_gpu", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--seg", type=str, default=None)

    parser.add_argument("--seg_gpu", type=str, default=None)
    parser.add_argument("--seg_step_per_epoch", type=int, default=None)
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    parser.add_argument("--seg_val_interval", type=int, default=None)

    parser.add_argument("--reg_gpu", type=str, default=None)
    parser.add_argument("--reg_step_per_epoch", type=int, default=None)
    parser.add_argument("--reg_checkpoint", type=str, default=None)
    parser.add_argument("--reg_val_interval", type=int, default=None)

    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_gpu", type=str, default=None)
    parser.add_argument("--save_image", action='store_true', default=None)
    parser.add_argument("--test_seg_checkpoint", type=str, default=None)
    parser.add_argument("--test_reg_checkpoint", type=str, default=None)

    parser.add_argument("--seg_lr", type=float, default=None)
    parser.add_argument("--seg_step_size", type=int, default=None)

    parser.add_argument("--reg_lr", type=float, default=None)
    parser.add_argument("--reg_step_size", type=int, default=None)

    parser.add_argument("--seg_model", type=str, default=None)
    parser.add_argument("--reg_model", type=str, default=None)

    parser.add_argument("--similarity_loss_use", type=bool, default=None)
    parser.add_argument("--similarity_loss_type", type=str, default=None)
    parser.add_argument("--similarity_loss_weight", type=float, default=None)

    parser.add_argument("--segmentation_loss_use", action='store_true', default=None)
    parser.add_argument("--segmentation_loss_type", type=str, default=None)
    parser.add_argument("--segmentation_loss_weight", type=float, default=None)

    parser.add_argument("--gradient_loss_use", action='store_true', default=None)
    parser.add_argument("--gradient_loss_weight", type=float, default=None)

    parser.add_argument("--supervised_loss_weight", type=float, default=None)
    parser.add_argument("--anatomy_loss_weight", type=float, default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    with open(args.config_path, "r", encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.train_data_path is not None:
        config['TrainConfig']['data_path'] = args.train_data_path
        if args.test_data_path is None:
            config['TestConfig']['data_path'] = args.train_data_path

    if args.train_gpu is not None:
        config['TrainConfig']['gpu'] = args.train_gpu

    if args.epoch is not None:
        config['TrainConfig']['epoch'] = args.epoch

    if args.batchsize is not None:
        config['TrainConfig']['batchsize'] = args.batchsize

    if args.seg is not None:
        config['TrainConfig']['seg'] = args.seg

    if args.seg_gpu is not None:
        config['TrainConfig']['Seg']['gpu'] = args.seg_gpu

    if args.seg_step_per_epoch is not None:
        config['TrainConfig']['Seg']['seg_step_per_epoch'] = args.seg_step_per_epoch

    if args.seg_checkpoint is not None:
        config['TrainConfig']['Seg']['checkpoint'] = args.seg_checkpoint

    if args.seg_val_interval is not None:
        config['TrainConfig']['Seg']['val_interval'] = args.seg_val_interval

    if args.reg_gpu is not None:
        config['TrainConfig']['Reg']['gpu'] = args.reg_gpu

    if args.reg_step_per_epoch is not None:
        config['TrainConfig']['Reg']['reg_step_per_epoch'] = args.reg_step_per_epoch

    if args.reg_checkpoint is not None:
        config['TrainConfig']['Reg']['checkpoint'] = args.reg_checkpoint

    if args.reg_val_interval is not None:
        config['TrainConfig']['Reg']['val_interval'] = args.reg_val_interval

    if args.test_data_path is not None:
        config['TestConfig']['data_path'] = args.test_data_path

    if args.test_gpu is not None:
        config['TestConfig']['gpu'] = args.test_gpu

    if args.save_image is not None:
        config['TestConfig']['save_image'] = args.save_image

    if args.test_seg_checkpoint is not None:
        config['TestConfig']['Seg']['checkpoint'] = args.test_seg_checkpoint

    if args.test_reg_checkpoint is not None:
        config['TestConfig']['Reg']['checkpoint'] = args.test_reg_checkpoint

    if args.seg_model is not None:
        config['ModelConfig']['Seg']['type'] = args.seg_model

    if args.reg_model is not None:
        config['ModelConfig']['Reg']['type'] = args.reg_model

    if args.seg_lr is not None:
        config['OptimConfig']['Seg']['optimizer']['params']['lr'] = args.seg_lr

    if args.seg_step_size is not None:
        config['OptimConfig']['Seg']['lr_scheduler']['params']['step_size'] = args.seg_step_size

    if args.reg_lr is not None:
        config['OptimConfig']['Reg']['optimizer']['params']['lr'] = args.reg_lr

    if args.reg_step_size is not None:
        config['OptimConfig']['Reg']['lr_scheduler']['params']['step_size'] = args.reg_step_size

    if args.similarity_loss_use is not None:
        config['LossConfig']['Reg']['similarity_loss']['use'] = args.similarity_loss_use

    if args.similarity_loss_type is not None:
        config['LossConfig']['Reg']['similarity_loss']['type'] = args.similarity_loss_type

    if args.similarity_loss_weight is not None:
        config['LossConfig']['Reg']['similarity_loss']['weight'] = args.similarity_loss_weight

    if args.segmentation_loss_use is not None:
        config['LossConfig']['Reg']['segmentation_loss']['use'] = args.segmentation_loss_use

    if args.segmentation_loss_type is not None:
        config['LossConfig']['Reg']['segmentation_loss']['type'] = args.segmentation_loss_type

    if args.segmentation_loss_weight is not None:
        config['LossConfig']['Reg']['segmentation_loss']['weight'] = args.segmentation_loss_weight

    if args.gradient_loss_use is True:
        config['LossConfig']['Reg']['gradient_loss']['use'] = args.gradient_loss_use

    if args.gradient_loss_weight is not None:
        config['LossConfig']['Reg']['gradient_loss']['weight'] = args.gradient_loss_weight

    if args.supervised_loss_weight is not None:
        config['LossConfig']['Seg']['supervised_loss']['weight'] = args.supervised_loss_weight

    if args.anatomy_loss_weight is not None:
        config['LossConfig']['Seg']['anatomy_loss']['weight'] = args.anatomy_loss_weight

    basedir = os.path.dirname(args.output_path)
    os.makedirs(basedir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data=config, stream=f, allow_unicode=True, sort_keys=False)
