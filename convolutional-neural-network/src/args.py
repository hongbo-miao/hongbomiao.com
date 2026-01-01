import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--should_download_original_data",
        default=False,
        type=bool,
        help="Auto download original data (default: False)",
    )
    return parser.parse_args()
