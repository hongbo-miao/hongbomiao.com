import logging

import duckdb
import pandas as pd


def create_people_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, 6),
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 28, 22],
            "city": ["New York", "London", "Paris", "Tokyo", "Berlin"],
        }
    )


def save_to_duckdb(df: pd.DataFrame, db_path: str) -> None:
    with duckdb.connect(db_path) as conn:
        conn.register("df", df)
        conn.execute("create table if not exists people as select * from df")


def query_from_duckdb(db_path: str, age: int) -> list[tuple[str, int]]:
    with duckdb.connect(db_path) as conn:
        return conn.execute(
            "select name, age from people where age > ?", [age]
        ).fetchall()


def main() -> None:
    db_path = "people.duckdb"

    df = create_people_dataframe()
    save_to_duckdb(df, db_path)

    res = query_from_duckdb(db_path, age=25)
    logging.info("People older than 25:")
    logging.info(res)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
