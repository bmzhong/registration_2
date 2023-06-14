import argparse
import os

import yaml


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--start_new_model", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--inv_label_loss_start_epoch", type=int)

    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--train_gpu", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--seg", type=str, default=None)

    parser.add_argument("--use_mean_teacher", type=str, default=None)
    parser.add_argument("--consistency", type=float, default=None)
    parser.add_argument("--consistency_loss_type", type=int, default=None)

    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_gpu", type=str, default=None)
    parser.add_argument("--save_image", action='store_true', default=None)

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mask_type", type=str, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)

    parser.add_argument("--similarity_loss_use", type=str, default=None)
    parser.add_argument("--similarity_loss_type", type=str, default=None)
    parser.add_argument("--similarity_loss_weight", type=float, default=None)

    parser.add_argument("--segmentation_loss_use", type=str, default=None)
    parser.add_argument("--segmentation_loss_type", type=str, default=None)
    parser.add_argument("--segmentation_loss_weight", type=float, default=None)

    parser.add_argument("--inv_label_loss_use", type=str, default=None)
    parser.add_argument("--inv_label_loss_type", type=str, default=None)
    parser.add_argument("--inv_label_loss_weight", type=float, default=None)

    parser.add_argument("--gradient_loss_use", type=str, default=None)
    parser.add_argument("--gradient_loss_weight", type=float, default=None)

    parser.add_argument("--bending_energy_loss_use", type=str, default=None)
    parser.add_argument("--bending_energy_loss_weight", type=float, default=None)

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

    if args.checkpoint is not None:
        config['TrainConfig']['checkpoint'] = args.checkpoint
        config['OptimConfig']['load_checkpoint'] = True
        config['TrainConfig']['start_new_model'] = False

    if args.inv_label_loss_start_epoch is not None:
        config['TrainConfig']['inv_label_loss_start_epoch'] = args.inv_label_loss_start_epoch

    if args.start_new_model is not None:
        if args.start_new_model == 'False':
            config['TrainConfig']['start_new_model'] = False
        elif args.start_new_model == 'True':
            config['TrainConfig']['start_new_model'] = True

    if args.epoch is not None:
        config['TrainConfig']['epoch'] = args.epoch

    if args.batchsize is not None:
        config['TrainConfig']['batchsize'] = args.batchsize

    if args.val_interval is not None:
        config['TrainConfig']['val_interval'] = args.val_interval

    if args.seg is not None:
        config['TrainConfig']['seg'] = args.seg

    if args.use_mean_teacher is not None:
        if args.use_mean_teacher == 'False':
            config['TrainConfig']['use_mean_teacher'] = False
        elif args.use_mean_teacher == 'True':
            config['TrainConfig']['use_mean_teacher'] = True
        else:
            raise Exception()

    if args.consistency is not None:
        config['TrainConfig']['consistency'] = args.consistency

    if args.consistency_loss_type is not None:
        config['TrainConfig']['consistency_loss_type'] = args.consistency_loss_type

    if args.test_data_path is not None:
        config['TestConfig']['data_path'] = args.test_data_path

    if args.test_gpu is not None:
        config['TestConfig']['gpu'] = args.test_gpu

    if args.save_image is not None:
        config['TestConfig']['save_image'] = args.save_image

    if args.model is not None:
        config['ModelConfig']['type'] = args.model

    if args.mask_type is not None:
        config['ModelConfig']['mask_type'] = args.mask_type

    if args.lr is not None:
        config['OptimConfig']['optimizer']['params']['lr'] = args.lr

    if args.step_size is not None:
        config['OptimConfig']['lr_scheduler']['params']['step_size'] = args.step_size

    if args.similarity_loss_use is not None:
        if args.similarity_loss_use == "True":
            config['LossConfig']['similarity_loss']['use'] = True
        elif args.similarity_loss_use == "False":
            config['LossConfig']['similarity_loss']['use'] = False
        else:
            raise Exception()

    if args.similarity_loss_type is not None:
        config['LossConfig']['similarity_loss']['type'] = args.similarity_loss_type

    if args.similarity_loss_weight is not None:
        config['LossConfig']['similarity_loss']['weight'] = args.similarity_loss_weight

    if args.segmentation_loss_use is not None:
        if args.segmentation_loss_use == "True":
            config['LossConfig']['segmentation_loss']['use'] = True
        elif args.segmentation_loss_use == "False":
            config['LossConfig']['segmentation_loss']['use'] = False
        else:
            raise Exception()

    if args.segmentation_loss_type is not None:
        config['LossConfig']['segmentation_loss']['type'] = args.segmentation_loss_type

    if args.segmentation_loss_weight is not None:
        config['LossConfig']['segmentation_loss']['weight'] = args.segmentation_loss_weight

    if args.inv_label_loss_type is not None:
        config['LossConfig']['inv_label_loss']['type'] = args.inv_label_loss_type

    if args.inv_label_loss_weight is not None:
        config['LossConfig']['inv_label_loss']['weight'] = args.inv_label_loss_weight

    if args.gradient_loss_use is not None:
        if args.gradient_loss_use == "True":
            config['LossConfig']['gradient_loss']['use'] = True
        elif args.gradient_loss_use == "False":
            config['LossConfig']['gradient_loss']['use'] = False
        else:
            raise Exception()

    if args.gradient_loss_weight is not None:
        config['LossConfig']['gradient_loss']['weight'] = args.gradient_loss_weight

    if args.bending_energy_loss_use is not None:
        if args.bending_energy_loss_use == "True":
            config['LossConfig']['bending_energy_loss']['use'] = True
        elif args.bending_energy_loss_use == "False":
            config['LossConfig']['bending_energy_loss']['use'] = False
        else:
            raise Exception()
    if args.bending_energy_loss_weight is not None:
        config['LossConfig']['bending_energy_loss']['weight'] = args.bending_energy_loss_weight

    basedir = os.path.dirname(args.output_path)
    os.makedirs(basedir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data=config, stream=f, allow_unicode=True, sort_keys=False)
