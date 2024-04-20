package com.hongbomiao

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, struct}
import org.apache.spark.sql.types.LongType
import za.co.absa.abris.avro.functions.to_avro
import za.co.absa.abris.config.{AbrisConfig, ToAvroConfig}

object IngestFromS3ToKafka {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
      .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
      )
      .getOrCreate()

    val motorValueSchemaConfig: ToAvroConfig =
      AbrisConfig.toConfluentAvro.downloadSchemaByLatestVersion
        .andTopicNameStrategy("hm.motor.avro")
        .usingSchemaRegistry(
          "http://confluent-schema-registry.hm-confluent-schema-registry.svc:8081"
        )

    val df = spark.readStream
      .format("delta")
      .load("s3a://hm-production-bucket/delta-tables/motor_data")
      .withColumn("timestamp", (col("timestamp") * 1000).cast(LongType))
      .select(to_avro(struct("*"), motorValueSchemaConfig).as("value"))

    val query = df.writeStream
      .format("kafka")
      .option(
        "kafka.bootstrap.servers",
        "hm-kafka-kafka-bootstrap.hm-kafka.svc:9092"
      )
      .option("topic", "hm.motor.avro")
      .option("checkpointLocation", "/tmp/checkpoint")
      .start()

    query.awaitTermination()
  }
}
