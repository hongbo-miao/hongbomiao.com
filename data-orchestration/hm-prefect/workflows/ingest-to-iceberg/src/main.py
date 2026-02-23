import os

from prefect import flow, get_run_logger, task
from pydantic import BaseModel
from pyspark.sql import SparkSession


class SparkConnectConfig(BaseModel):
    spark_connect_url: str
    parquet_data_path: str


@task(name="Ingest to Iceberg")
def ingest_to_iceberg(spark_connect_url: str, parquet_data_path: str) -> None:
    logger = get_run_logger()

    spark = SparkSession.builder.remote(spark_connect_url).getOrCreate()

    iceberg_namespace = "emergency"
    iceberg_table_name = f"iceberg.{iceberg_namespace}.audio_segments"

    spark.sql(f"create namespace if not exists iceberg.{iceberg_namespace}")

    logger.info(f"Reading parquet files from {parquet_data_path}")
    data_frame = spark.read.parquet(parquet_data_path)

    row_count = data_frame.count()
    logger.info(f"Ingesting {row_count} rows to {iceberg_table_name}")

    data_frame.writeTo(iceberg_table_name).using("iceberg").createOrReplace()

    logger.info(
        "Successfully wrote parquet data to Iceberg table via Spark Connect",
    )


@flow
def hm_ingest_to_iceberg(config: SparkConnectConfig) -> None:
    ingest_to_iceberg(config.spark_connect_url, config.parquet_data_path)


if __name__ == "__main__":
    spark_connect_config = SparkConnectConfig(
        spark_connect_url=os.environ.get(
            "SPARK_CONNECT_URL",
            "sc://localhost:15002",
        ),
        parquet_data_path=os.environ.get(
            "PARQUET_DATA_PATH",
            "s3a://iceberg-bucket/data/",
        ),
    )
    hm_ingest_to_iceberg(spark_connect_config)
