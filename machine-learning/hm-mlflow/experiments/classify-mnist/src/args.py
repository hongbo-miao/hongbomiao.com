import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5)
    return parser.parse_args()
