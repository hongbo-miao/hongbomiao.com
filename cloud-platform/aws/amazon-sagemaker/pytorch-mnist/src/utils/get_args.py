import argparse
import json
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # SageMaker container environments
    parser.add_argument(
        "--hosts",
        type=list,
        default=json.loads(os.environ["SM_HOSTS"]),
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=os.environ["SM_CURRENT_HOST"],
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=os.environ["SM_NUM_GPUS"],
    )

    return parser.parse_args()
