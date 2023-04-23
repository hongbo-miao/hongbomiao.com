package com.hongbomiao

import org.apache.spark.sql.{DataFrame, SparkSession}
import com.typesafe.config.ConfigFactory

object IngestFromS3ToKafka {
  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.load("application-development.conf")
    val awsAccessKeyID = config.getString("aws-access-key-id")
    val awsSecretAccessKey = config.getString("aws-secret-access-key")

    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("ingest-from-s3-to-kafka")
      .config("spark.ui.port", "4040")
      .config("spark.hadoop.fs.s3a.access.key", awsAccessKeyID)
      .config("spark.hadoop.fs.s3a.secret.key", awsSecretAccessKey)
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
