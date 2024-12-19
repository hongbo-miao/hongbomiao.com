import logging
from pathlib import Path
import httpx
import polars as pl


def download_data() -> None:
    transaction_url = "https://storage.googleapis.com/rapidsai/polars-demo/transactions-t4-20.parquet"
    transaction_file = Path("transactions.parquet")
    
    if not transaction_file.exists():
        with httpx.Client() as client:
            response = client.get(transaction_url)
            response.raise_for_status()
            transaction_file.write_bytes(response.content)


def demo_gpu_operations() -> None:
    # Download data if not exists
    download_data()
    
    # Load transaction data
    transactions = pl.scan_parquet("transactions.parquet")
    
    # Configure GPU engine with error reporting
    gpu_engine = pl.GPUEngine(
        device=0,
        raise_on_fail=True,  # Fail loudly if we can't run on the GPU
    )

    # Calculate total transaction amount using GPU engine
    total_amount = (
        transactions
        .select(pl.col("AMOUNT").sum().alias("total_amount"))
        .collect(engine=gpu_engine)
    )
    logging.info(f"Total transaction amount: {total_amount}")

    # Group by category and calculate statistics
    category_stats = (
        transactions
        .groupby("CATEGORY")
        .agg(
            pl.col("AMOUNT").sum().alias("total_amount"),
            pl.col("AMOUNT").mean().alias("avg_amount"),
            pl.col("AMOUNT").count().alias("transaction_count"),
        )
        .collect(engine=gpu_engine)
    )
    logging.info("\nCategory statistics:")
    logging.info(category_stats)


def main() -> None:
    demo_gpu_operations()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
