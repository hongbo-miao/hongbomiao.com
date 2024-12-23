import logging
from pathlib import Path

import duckdb
import pandas as pd


def create_people_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 28, 22],
            "city": ["New York", "London", "Paris", "Tokyo", "Berlin"],
        }
    )


def save_to_duckdb(df: pd.DataFrame, db_path: Path) -> None:
    with duckdb.connect(db_path) as conn:
        conn.register("df", df)
        conn.execute("create table if not exists people as select * from df")


def query_from_duckdb(db_path: Path, age: int) -> pd.DataFrame:
    with duckdb.connect(db_path) as conn:
        return conn.execute("select name, age from people where age > ?", [age]).df()


def main() -> None:
    db_path = Path("data/people.duckdb")

    df = create_people_dataframe()
    logging.info(df)
    save_to_duckdb(df, db_path)

    df = query_from_duckdb(db_path, age=25)
    logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
