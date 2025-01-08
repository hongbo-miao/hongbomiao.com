import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin-virtual",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0.5,
        help="dropout ratio (default: 0.5)",
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=300,
        help="dimensionality of hidden units in GNNs (default: 300)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        help="dataset name (default: ogbg-molhiv)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="full",
        help="full feature or simple feature",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="filename to output result (default: )",
    )
    return parser.parse_args()
