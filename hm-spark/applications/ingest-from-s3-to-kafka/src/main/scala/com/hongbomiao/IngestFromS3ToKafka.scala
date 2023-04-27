package com.hongbomiao

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.to_json
import org.apache.spark.sql.types.{DoubleType, StructType}

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

    // val schema = spark.read.parquet("s3a://hongbomiao-bucket/iot/motor.parquet").schema
    val schema = new StructType()
      .add("timestamp", DoubleType)
      .add("current", DoubleType)
      .add("voltage", DoubleType)
      .add("temperature", DoubleType)

    val df = spark.readStream
      .schema(schema)
      .option("maxFilesPerTrigger", 1)
      .parquet(folderPath)
      .select(to_json(struct("*")).alias("value"))

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
