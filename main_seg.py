import argparse
import os
import time
import warnings
from util.util import set_random_seed, get_basedir
from shutil import copyfile
import yaml
from train_seg import train_seg
from test_seg import test_seg

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", "-t", action="store_true",
                        help="train mode, you must give the --output and --config")
    parser.add_argument("--test", action="store_true",
                        help="test mode, you must give the --output and --config")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='if the mode is train: the dir to store the file;'
                             'if the mode is eval or ave: the path of output')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='used in all the modes, the path of the config yaml')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='used in the test mode, the path of the checkpoint')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(seed=0)
    with open(args.config, "r", encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    basedir = get_basedir(
        args.output, config["TrainConfig"]["start_new_model"])

    copyfile(args.config, os.path.join(basedir, "config.yaml"))
    print(f"base dir is {basedir}")
    start_time = time.time()
    if args.train:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["TrainConfig"]["gpu"]
        train_seg(config, basedir)

        checkpoint = os.path.join(basedir, "checkpoint", 'best_epoch.pth')
        test_basedir = get_basedir(os.path.join(basedir, 'test'), config["TrainConfig"]["start_new_model"])
        test_seg(config, test_basedir, checkpoint=checkpoint)

    elif args.test:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["TestConfig"]["gpu"]
        test_seg(config, basedir, args.checkpoint)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"run time: {int(run_time)} s")