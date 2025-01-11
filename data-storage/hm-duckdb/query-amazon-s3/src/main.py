import logging
from pathlib import Path

import duckdb
import xxhash
from config import Config
from utils.clean_table_name import clean_table_name
from utils.get_file_true_stem import get_file_true_stem

logger = logging.getLogger(__name__)


def get_cache_table_name(parquet_url: str) -> str:
    url_hash = xxhash.xxh128(parquet_url.encode()).hexdigest()
    table_name = clean_table_name(get_file_true_stem(Path(parquet_url.split("/")[-1])))
    return f"{table_name}_{url_hash}"


def main(parquet_url: str) -> None:
    config = Config()
    duckdb_cache_db_path = Path("data/cache.duckdb")
    logger.info(f"Using DuckDB cache file: {duckdb_cache_db_path}")

    with duckdb.connect(duckdb_cache_db_path) as conn:
        try:
            # Configure DuckDB settings
            conn.execute("set enable_progress_bar=true")
            conn.execute(
                f"""
                    create secret if not exists http_auth (
                        type http,
                        bearer_token '{config.HTTP_AUTH_TOKEN}'
                    )
                """,
            )

            # Create DuckDB cache
            logger.info("Loading data...")
            table_name = get_cache_table_name(parquet_url)
            conn.execute(f"""
                create table if not exists {table_name} as
                select * from read_parquet('{parquet_url}')
            """)  # noqa: S608

            # Query
            query = f"""
                select *
                from {table_name}
            """  # noqa: S608

            df = conn.execute(query).pl()
            logger.info(df)
        except Exception:
            logger.exception("An error occurred")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parquet_url = (
        "https://data-browser.internal.hongbomiao.com/experiments/experiment1.parquet"
    )
    main(parquet_url)
