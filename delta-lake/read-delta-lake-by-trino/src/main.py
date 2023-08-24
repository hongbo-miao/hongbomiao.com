import logging

import config
import pandas as pd

import trino


def main():
    conn = trino.dbapi.connect(
        host=config.trino_host, port=config.trino_port, user=config.trino_user
    )
    query = "select * from delta.hm_iot_db.motor limit 100"
    df = pd.read_sql_query(query, conn)
    logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
