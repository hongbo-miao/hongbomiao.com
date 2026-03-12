import logging

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

SPARK_CONNECT_URL = "sc://spark-connect.hongbomiao.com:443/;use_ssl=true"


def main() -> None:
    spark = SparkSession.builder.remote(SPARK_CONNECT_URL).getOrCreate()

    logger.info("Connected to Spark Connect server")
    logger.info(f"Spark version: {spark.version}")

    logger.info("Creating namespace 'random' in development catalog if not exists:")
    spark.sql("create namespace if not exists development.random")

    logger.info("Listing namespaces in development catalog:")
    spark.sql("show namespaces in development").show()

    logger.info("Writing sample data to development.random.people:")
    sample_data = [(1, "Alice", 30), (2, "Bob", 25), (3, "Charlie", 35)]
    column_names = ["id", "name", "age"]
    data_frame = spark.createDataFrame(sample_data, column_names)
    data_frame.writeTo("development.random.people").createOrReplace()

    logger.info("Reading back from development.random.people:")
    spark.table("development.random.people").show()

    logger.info("Running aggregation:")
    spark.table("development.random.people").selectExpr(
        "avg(age) as average_age",
        "max(age) as max_age",
    ).show()

    spark.stop()
    logger.info("Disconnected from Spark Connect server")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
