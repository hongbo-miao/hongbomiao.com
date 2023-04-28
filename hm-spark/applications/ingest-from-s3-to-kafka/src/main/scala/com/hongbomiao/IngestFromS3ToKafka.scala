package com.hongbomiao

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{array, col, lit, struct, to_json}
import org.apache.spark.sql.types.{
  DecimalType,
  DoubleType,
  LongType,
  StructType
}

object IngestFromS3ToKafka {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("ingest-from-s3-to-kafka")
      .config("spark.ui.port", "4040")
      // .config("spark.hadoop.fs.s3a.access.key", "xxx")
      // .config("spark.hadoop.fs.s3a.secret.key", "xxx")
      .getOrCreate()

    val folderPath = "s3a://hongbomiao-bucket/iot/"

    // val parquet_schema = spark.read.parquet("s3a://hongbomiao-bucket/iot/motor.parquet").schema
    val parquet_schema = new StructType()
      .add("timestamp", DoubleType)
      .add("current", DoubleType, nullable = true)
      .add("voltage", DoubleType, nullable = true)
      .add("temperature", DoubleType, nullable = true)

    val df = spark.readStream
      .schema(parquet_schema)
      .option("maxFilesPerTrigger", 8)
      .parquet(folderPath)
      .withColumn("timestamp", (col("timestamp") * 1000).cast(LongType))
      .select(
        to_json(
          struct(
            struct(
              lit("struct").alias("type"),
              array(
                struct(
                  lit("int64").alias("type"),
                  lit(false).alias("optional"),
                  lit("timestamp").alias("field")
                ),
                struct(
                  lit("double").alias("type"),
                  lit(true).alias("optional"),
                  lit("current").alias("field")
                ),
                struct(
                  lit("double").alias("type"),
                  lit(true).alias("optional"),
                  lit("voltage").alias("field")
                ),
                struct(
                  lit("double").alias("type"),
                  lit(true).alias("optional"),
                  lit("temperature").alias("field")
                )
              ).alias("fields")
            ).alias("schema"),
            struct("*").alias("payload")
          )
        ).alias("value")
      )

    val query = df.writeStream
      .format("kafka")
      .option(
        "kafka.bootstrap.servers",
        "hm-kafka-kafka-bootstrap.hm-kafka.svc:9092"
      )
      .option("topic", "hm.motor")
      .option("checkpointLocation", "/tmp/checkpoint")
      .start()

    query.awaitTermination()
  }
}
