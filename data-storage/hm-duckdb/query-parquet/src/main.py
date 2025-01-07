import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


def main(parquet_path: Path) -> None:
    with duckdb.connect() as conn:
        query = f"""
            select *
            from read_parquet('{parquet_path}')
        """
        df = conn.execute(query).pl()
        logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parquet_path = Path("data/file.parquet")
    main(parquet_path)
