import logging
from pathlib import Path

import duckdb
import polars as pl


def main(duckdb_path: Path) -> None:
    people = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 28, 22],
            "city": ["New York", "London", "Paris", "Tokyo", "Berlin"],
        }
    )
    logging.info(people)
    with duckdb.connect(duckdb_path) as conn:
        conn.register("people", people)
        conn.execute("create table if not exists people as select * from people")

        age = 25
        df = conn.execute("select name, age from people where age > ?", [age]).pl()
        logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    duckdb_path = Path("data/people.duckdb")
    main(duckdb_path)
