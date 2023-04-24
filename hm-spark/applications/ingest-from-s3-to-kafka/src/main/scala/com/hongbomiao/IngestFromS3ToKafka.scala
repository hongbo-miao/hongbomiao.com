package com.hongbomiao

import org.apache.spark.sql.{DataFrame, SparkSession}

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

    import spark.implicits._

    val filePath = "s3a://hongbomiao-bucket/dc-motor/EHM.parquet"
    val df = spark.read.parquet(filePath)
    df.createOrReplaceTempView("ehm")

    val retiredPeople: DataFrame = spark.sql("SELECT * FROM ehm")
    retiredPeople.show()

    spark.stop()
  }
}
