name := "IngestFromS3ToKafka"
version := "1.0"
scalaVersion := "2.12.17"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.2" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.3.2" % "provided",
  "org.apache.spark" %% "spark-sql-kafka-0-10" % "3.3.2" % "provided",
  "org.apache.spark" %% "spark-avro" % "3.4.0" % "provided",
  "org.apache.hadoop" % "hadoop-common" % "3.3.5" % "provided",
  "org.apache.hadoop" % "hadoop-aws" % "3.3.5" % "provided",
  "com.amazonaws" % "aws-java-sdk-bundle" % "1.12.463" % "provided",
  "com.softwaremill.sttp.client3" %% "core" % "3.8.15"
)
