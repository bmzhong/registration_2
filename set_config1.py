import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", "-t", action="store_true",
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