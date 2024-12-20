import logging
import tempfile
import time
from pathlib import Path

import httpx
import pandas as pd


def download_data(transaction_path: Path) -> None:
    transaction_url = (
        "https://storage.googleapis.com/rapidsai/polars-demo/transactions-t4-20.parquet"
    )

    if not transaction_path.exists():
        with httpx.Client() as client:
            response = client.get(transaction_url)
            response.raise_for_status()
            transaction_path.write_bytes(response.content)


def main() -> None:
    # Download data if not exists
    transaction_path = Path(tempfile.gettempdir()) / "transactions.parquet"
    print(f"File location: {transaction_path}")
    download_data(transaction_path)

    # Time the data loading
    start_time = time.time()
    transactions = pd.read_parquet(transaction_path)
    load_time = time.time() - start_time
    logging.info(f"Data loading time: {load_time:.2f} seconds")

    # Print DataFrame info
    logging.info(transactions.columns)
    logging.info(transactions.head())

    # Time the sum calculation
    start_time = time.time()
    total_amount = transactions["AMOUNT"].sum()
    sum_time = time.time() - start_time
    logging.info(f"Total transaction amount: {total_amount}")
    logging.info(f"Sum calculation time: {sum_time:.2f} seconds")

    # Time the group by operations
    start_time = time.time()
    category_stats = (
        transactions.groupby("EXP_TYPE")
        .agg({"AMOUNT": ["sum", "mean", "count"]})
        .rename(
            columns={
                "AMOUNT_sum": "total_amount",
                "AMOUNT_mean": "avg_amount",
                "AMOUNT_count": "transaction_count",
            }
        )
    )
    group_by_time = time.time() - start_time
    logging.info("Category statistics:")
    logging.info(category_stats)
    logging.info(f"Group by operations time: {group_by_time:.2f} seconds")

    # Print total execution time
    logging.info(
        f"Total execution time: {load_time + sum_time + group_by_time:.2f} seconds"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
