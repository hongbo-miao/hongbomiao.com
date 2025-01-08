import logging

import awswrangler as wr

logger = logging.getLogger(__name__)


def main() -> None:
    df = wr.athena.read_sql_query(
        "select * from motor limit 10;",
        database="production_hm_iot_db",
    )
    logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
