import logging
import tempfile
import time
from pathlib import Path

import httpx
import polars as pl


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

    gpu_engine = pl.GPUEngine(
        device=0,
        raise_on_fail=True,  # Fail loudly if we cannot run on the GPU.
    )

    # Time the data loading
    start_time = time.time()
    transactions = pl.scan_parquet(transaction_path)
    load_time = time.time() - start_time
    logging.info(f"Data loading time: {load_time:.2f} seconds")

    # Print DataFrame info
    logging.info(transactions.collect_schema().names())
    logging.info(transactions.head())

    # Time the sum calculation
    start_time = time.time()
    total_amount = transactions.select(pl.col("AMOUNT").sum()).collect(engine="gpu")
    sum_time = time.time() - start_time
    logging.info(f"Total transaction amount: {total_amount}")
    logging.info(f"Sum calculation time: {sum_time:.2f} seconds")

    # Time the group by operations
    start_time = time.time()
    category_stats = (
        transactions.group_by("EXP_TYPE")
        .agg(
            [
                pl.col("AMOUNT").sum().alias("total_amount"),
                pl.col("AMOUNT").mean().alias("avg_amount"),
                pl.col("AMOUNT").count().alias("transaction_count"),
            ]
        )
        .collect(engine=gpu_engine)
    )
    group_by_time = time.time() - start_time
    logging.info("Category statistics:")
    logging.info(category_stats)
    logging.info(f"Groupby operations time: {group_by_time:.2f} seconds")

    # Print total execution time
    logging.info(
        f"Total execution time: {load_time + sum_time + group_by_time:.2f} seconds"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
