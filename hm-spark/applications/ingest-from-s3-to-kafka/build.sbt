name := "IngestFromS3ToKafka"
version := "1.0"
scalaVersion := "2.12.17"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.4.0",
  "org.apache.spark" %% "spark-sql" % "3.4.0",
  "org.apache.hadoop" % "hadoop-common" % "3.3.5",
  "org.apache.hadoop" % "hadoop-aws" % "3.3.5",
  "com.amazonaws" % "aws-java-sdk-bundle" % "1.12.454",
  "com.typesafe" % "config" % "1.4.2",
)
