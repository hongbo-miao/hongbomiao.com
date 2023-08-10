import config
import pandas as pd

import trino


def main():
    conn = trino.dbapi.connect(
        host=config.trino_host,
        port=config.trino_port,
        user=config.trino_user,
        catalog="delta",
        schema="hm_iot_db",
    )
    query = "SELECT * FROM motor LIMIT 10"
    res = pd.read_sql_query(query, conn)
    print(res)


if __name__ == "__main__":
    main()
