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
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--seg", type=str, default=None)

    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--test_gpu", type=str, default=None)
    parser.add_argument("--save_image", action='store_true', default=None)

    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)

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

    if args.val_interval is not None:
        config['TrainConfig']['val_interval'] = args.val_interval

    if args.seg is not None:
        config['TrainConfig']['seg'] = args.seg

    if args.test_data_path is not None:
        config['TestConfig']['data_path'] = args.test_data_path

    if args.test_gpu is not None:
        config['TestConfig']['gpu'] = args.test_gpu

    if args.save_image is not None:
        config['TestConfig']['save_image'] = args.save_image

    if args.model is not None:
        config['ModelConfig']['type'] = args.model

    if args.lr is not None:
        config['OptimConfig']['optimizer']['params']['lr'] = args.lr

    if args.step_size is not None:
        config['OptimConfig']['lr_scheduler']['params']['step_size'] = args.step_size


    basedir = os.path.dirname(args.output_path)
    os.makedirs(basedir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data=config, stream=f, allow_unicode=True, sort_keys=False)
