package com.hongbomiao

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.to_json

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

    val filePath = "s3a://hongbomiao-bucket/sensor/EHM.parquet"
    spark.read
      .parquet(filePath)
      .select(to_json(struct("*")).alias("value"))
      .write
      .format("kafka")
      .option(
        "kafka.bootstrap.servers",
        "hm-kafka-kafka-bootstrap.hm-kafka.svc:9092"
      )
      .option("topic", "hm.ehm")
      .save()

    spark.stop()
  }
}
