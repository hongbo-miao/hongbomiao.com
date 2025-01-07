import sqlite3

import pandas as pd

con = sqlite3.connect("driver_features/data/online_store.db")
print("\n--- Schema of online store ---")
print(
    pd.read_sql_query(
        "SELECT * FROM driver_features_driver_hourly_stats",
        con,
    ).columns.tolist(),
)
con.close()
