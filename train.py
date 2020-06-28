import argparse
import torch


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch', type=int, default=8)
    return parser.parse_args()


def _main():
    args = add_arguments()
    print(args)


if __name__ == '__main__':
    _main()
