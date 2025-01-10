from pyspark.sql import DataFrame, SparkSession
from utils.df import union_all


def load_trips(spark: SparkSession, data_paths: list[str]) -> DataFrame:
    trip_dfs = [spark.read.parquet(data_path) for data_path in data_paths]
    return union_all(*trip_dfs)


def preprocess_trips(df: DataFrame) -> DataFrame:
    column_names = [x.lower() for x in df.columns]
    return df.toDF(*column_names)
