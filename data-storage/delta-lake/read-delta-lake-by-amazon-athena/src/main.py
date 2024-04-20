import logging

import awswrangler as wr


def main():
    df = wr.athena.read_sql_query(
        "select * from motor limit 10;",
        database="production_hm_iot_db",
    )
    logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
