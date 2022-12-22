from pyspark.sql import DataFrame, SparkSession
from utils.df import union_all


def load_trips(spark: SparkSession, data_paths: list[str]):
    trip_dfs = []
    for path in data_paths:
        trip_dfs.append(spark.read.parquet(path))
    return union_all(*trip_dfs)


def preprocess_trips(df: DataFrame) -> DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    return df.toDF(*column_names)
