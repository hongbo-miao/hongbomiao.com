import logging
from pathlib import Path

import duckdb


def main(parquet_path: Path) -> None:
    with duckdb.connect() as conn:
        query = f"""
            select *
            from read_parquet('{parquet_path}')
        """
        df = conn.execute(query).df()
        logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parquet_path = Path("data/file.parquet")
    main(parquet_path)
