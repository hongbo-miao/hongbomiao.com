import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812

logger = logging.getLogger(__name__)

SPARK_CONNECT_URL = "sc://spark-connect-large.hongbomiao.com:443/;use_ssl=true"
CHECKPOINT_BASE_PATH = "s3a://production-hm-spark-checkpoints"
SOURCE_PATH = "s3a://production-hm-data-raw/nats/motor/"
CATALOG = "production"
NAMESPACE = "motor_db"
TABLE_NAME = "motor_data"
PARTITION_COLUMN = "_time"


def create_partitioned_table_if_not_exists(
    spark: SparkSession,
    table_name: str,
) -> None:
    # Partition by hour.
    # _time is timestamptz (microsecond precision) derived from timestamp_ns.
    spark.sql(
        f"create table if not exists {table_name} "
        f"(`{PARTITION_COLUMN}` timestamp) "
        f"using iceberg "
        f"partitioned by (hours({PARTITION_COLUMN}))",
    )


def main() -> None:
    spark = SparkSession.builder.remote(SPARK_CONNECT_URL).getOrCreate()
    logger.info("Created Spark Connect session")

    logger.info("Connecting to Spark Connect server")
    logger.info(f"Spark version: {spark.version}")
    logger.info("Connected to Spark Connect server")

    logger.info(f"Creating namespace {CATALOG}.{NAMESPACE} if not exists")
    spark.sql(f"create namespace if not exists {CATALOG}.{NAMESPACE}")

    logger.info(f"Inferring merged schema from {SOURCE_PATH}")
    source_schema = spark.read.option("mergeSchema", "true").parquet(SOURCE_PATH).schema

    logger.info(
        f"Creating partitioned table {CATALOG}.{NAMESPACE}.{TABLE_NAME} if not exists",
    )
    full_table_name = f"{CATALOG}.{NAMESPACE}.{TABLE_NAME}"
    create_partitioned_table_if_not_exists(spark, full_table_name)

    logger.info(f"Streaming parquet files from {SOURCE_PATH} to {full_table_name}")
    streaming_data_frame = (
        spark.readStream.schema(source_schema)
        .parquet(SOURCE_PATH)
        .withColumn(
            PARTITION_COLUMN,
            (F.col("timestamp_ns") / 1_000_000_000).cast("timestamp"),
        )
        .select(PARTITION_COLUMN, *source_schema.fieldNames())
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
            f"{CHECKPOINT_BASE_PATH}/{CATALOG}/{NAMESPACE}/{TABLE_NAME}",
        )
        # Scheduled incremental batch processing. Processes as much data as is available at the time the streaming job is triggered.
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
