import logging
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)

con = sqlite3.connect("driver_features/data/online_store.db")
logger.info("--- Schema of online store ---")
logger.info(
    pd.read_sql_query(
        "SELECT * FROM driver_features_driver_hourly_stats",
        con,
    ).columns.tolist(),
)
con.close()
