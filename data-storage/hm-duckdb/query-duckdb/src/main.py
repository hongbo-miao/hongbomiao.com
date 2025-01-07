import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


def main(duckdb_path: Path) -> None:
    people = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 28, 22],
            "city": ["New York", "London", "Paris", "Tokyo", "Berlin"],
        },
    )
    logger.info(people)
    with duckdb.connect(duckdb_path) as conn:
        conn.register("people", people)
        conn.execute("create table if not exists people as select * from people")

        age = 25
        df = conn.execute("select name, age from people where age > ?", [age]).pl()
        logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    duckdb_path = Path("data/people.duckdb")
    main(duckdb_path)
