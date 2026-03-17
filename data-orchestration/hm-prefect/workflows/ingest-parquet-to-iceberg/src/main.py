import os

from prefect import flow, get_run_logger, task
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812


class IngestParquetToIcebergConfig(BaseModel):
    spark_connect_url: str
    checkpoint_base_path: str
    source_path: str
    catalog: str
    namespace: str
    table_name: str
    partition_column: str


@task
def ingest_parquet_to_iceberg(
    spark_connect_url: str,
    checkpoint_base_path: str,
    source_path: str,
    catalog: str,
    namespace: str,
    table_name: str,
    partition_column: str,
) -> None:
    logger = get_run_logger()

    spark = SparkSession.builder.remote(spark_connect_url).getOrCreate()
    logger.info("Created Spark Connect session")
    logger.info(f"Spark version: {spark.version}")

    logger.info(f"Creating namespace {catalog}.{namespace} if not exists")
    spark.sql(f"create namespace if not exists {catalog}.{namespace}")

    logger.info(f"Inferring merged schema from {source_path}")
    source_schema = spark.read.option("mergeSchema", "true").parquet(source_path).schema

    full_table_name = f"{catalog}.{namespace}.{table_name}"
    logger.info(f"Creating partitioned table {full_table_name} if not exists")
    # Partition by hour.
    # _time is timestamptz (microsecond precision) derived from timestamp_ns (bigint, nanosecond precision)
    spark.sql(
        f"create table if not exists {full_table_name} "
        f"(`{partition_column}` timestamp) "
        f"using iceberg "
        f"partitioned by (hours({partition_column}))",
    )

    logger.info(f"Streaming parquet files from {source_path} to {full_table_name}")
    streaming_data_frame = (
        spark.readStream.schema(source_schema)
        .parquet(source_path)
        .withColumn(
            partition_column,
            (F.col("timestamp_ns") / 1_000_000_000).cast("timestamp"),
        )
        .select(partition_column, *source_schema.fieldNames())
    )
    query = (
        streaming_data_frame.writeStream.format("iceberg")
        .outputMode("append")
        .option("mergeSchema", "true")
        .option("write-format", "parquet")
        .option("compression-codec", "zstd")
        .option("compression-level", "19")
        .option(
            "checkpointLocation",
            f"{checkpoint_base_path}/{catalog}/{namespace}/{table_name}",
        )
        # availableNow=True processes all new files since last checkpoint, then stops
        .trigger(availableNow=True)
        .toTable(full_table_name)
    )
    query.awaitTermination()

    logger.info(f"Reading back from {full_table_name}")
    iceberg_data_frame = spark.table(full_table_name)
    iceberg_row_count = iceberg_data_frame.count()
    logger.info(f"Iceberg table has {iceberg_row_count} rows")
    iceberg_data_frame.show(20, truncate=False)

    spark.stop()
    logger.info("Disconnected from Spark Connect server")


@flow
def hm_ingest_parquet_to_iceberg(config: IngestParquetToIcebergConfig) -> None:
    ingest_parquet_to_iceberg(
        config.spark_connect_url,
        config.checkpoint_base_path,
        config.source_path,
        config.catalog,
        config.namespace,
        config.table_name,
        config.partition_column,
    )


if __name__ == "__main__":
    ingest_parquet_to_iceberg_config = IngestParquetToIcebergConfig(
        spark_connect_url=os.environ.get("SPARK_CONNECT_URL"),
        checkpoint_base_path=os.environ.get("CHECKPOINT_BASE_PATH"),
        source_path=os.environ.get("SOURCE_PATH"),
        catalog=os.environ.get("CATALOG"),
        namespace=os.environ.get("NAMESPACE"),
        table_name=os.environ.get("TABLE_NAME"),
        partition_column=os.environ.get("PARTITION_COLUMN"),
    )
    hm_ingest_parquet_to_iceberg(ingest_parquet_to_iceberg_config)
