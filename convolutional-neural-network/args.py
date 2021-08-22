import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    return parser.parse_args()
