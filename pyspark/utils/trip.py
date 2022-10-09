from utils.df import unionAll

from pyspark.sql import DataFrame


def load_trips(spark, data_paths):
    trip_dfs = []
    for path in data_paths:
        trip_dfs.append(spark.read.parquet(path))
    return unionAll(*trip_dfs)


def preprocess_trips(df: DataFrame) -> DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    return df.toDF(*column_names)
