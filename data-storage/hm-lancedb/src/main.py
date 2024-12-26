import logging

import lancedb
import polars as pl
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

registry = get_registry()
func = registry.get("sentence-transformers").create(name="all-MiniLM-L6-v2")


class Quotes(LanceModel):
    name: str
    line: str = func.SourceField()
    vector: Vector = func.VectorField()


def create_and_populate_table(
    db: lancedb.connect, df: pl.DataFrame
) -> lancedb.table.Table:
    table = db.create_table("quotes", schema=Quotes, mode="overwrite")
    table.add(df)
    return table


def perform_semantic_search(table: lancedb.table.Table, query: str) -> pl.DataFrame:
    return table.search(query).limit(5).to_polars()


def main():
    url = "https://raw.githubusercontent.com/Abhiram970/RickBot/refs/heads/main/Rick_and_Morty.csv"
    df = pl.read_csv(url)
    db = lancedb.connect("~/.lancedb")
    table = create_and_populate_table(db, df)

    query = "What is the meaning of life?"
    df = perform_semantic_search(table, query)
    logging.info("Question: %s", query)
    logging.info("Answer: %s", df["line"][0])
    logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
