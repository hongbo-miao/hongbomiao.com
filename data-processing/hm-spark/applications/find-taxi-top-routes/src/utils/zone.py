from pyspark.sql import DataFrame, SparkSession


def load_zones(spark: SparkSession, zone_data_path: str) -> DataFrame:
    return (
        spark.read.format("csv")
        .option("inferSchema", True)  # noqa: FBT003
        .option("header", True)  # noqa: FBT003
        .load(zone_data_path)
    )


def preprocess_zones(df: DataFrame) -> DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    df = df.toDF(*column_names)
    return df.drop("objectid")
