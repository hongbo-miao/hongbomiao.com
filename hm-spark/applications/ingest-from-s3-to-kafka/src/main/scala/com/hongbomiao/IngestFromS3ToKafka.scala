package com.hongbomiao

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, struct}
import org.apache.spark.sql.types.{DoubleType, LongType, StructType}
import org.apache.spark.sql.avro.functions.to_avro
import sttp.client3.{HttpClientSyncBackend, UriContext, basicRequest}

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
    val parquetSchema = new StructType()
      .add("timestamp", DoubleType)
      .add("current", DoubleType, nullable = true)
      .add("voltage", DoubleType, nullable = true)
      .add("temperature", DoubleType, nullable = true)

    val backend = HttpClientSyncBackend()
    val res = basicRequest
      .get(
        uri"http://apicurio-registry-apicurio-registry.hm-apicurio-registry.svc:8080/apis/registry/v2/groups/hm-group/artifacts/hm.motor-value"
      )
      .send(backend)
    val kafkaRecordValueSchema = res.body.fold(identity, identity)

    val df = spark.readStream
      .schema(parquetSchema)
      .option("maxFilesPerTrigger", 1)
      .parquet(folderPath)
      .withColumn("timestamp", (col("timestamp") * 1000).cast(LongType))
      .select(to_avro(struct("*"), kafkaRecordValueSchema).alias("value"))

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
