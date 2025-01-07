import logging
from pathlib import Path

import duckdb
import lancedb

logger = logging.getLogger(__name__)


def main(database_path: Path) -> None:
    sample_data = [
        {
            "product_name": "LED Bulb",
            "product_price": 10.0,
            "review_scores": [4.5, 4.0, 4.8, 5.0, 4.1],
        },
        {
            "product_name": "Power Bank",
            "product_price": 20.0,
            "review_scores": [3.5, 4.0, 4.1],
        },
    ]
    db = lancedb.connect(str(database_path))
    product_table = db.create_table(
        "product_catalog",
        data=sample_data,
        mode="overwrite",
    ).to_lance()

    with duckdb.connect() as conn:
        conn.register("product_table", product_table)
        query = """
            select
                product_name,
                product_price,
                list_avg(review_scores) as average_review_score
            from product_table
            order by average_review_score desc
        """
        df = conn.execute(query).pl()
        logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    database_path = Path("/tmp/lancedb/products")
    main(database_path)
