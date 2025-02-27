import argparse
from argparse import Namespace


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str)
    return parser.parse_args()
