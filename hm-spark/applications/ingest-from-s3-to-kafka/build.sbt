name := "IngestFromS3ToKafka"
version := "1.0"
scalaVersion := "2.12.17"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.2" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.3.2" % "provided",
  "org.apache.hadoop" % "hadoop-common" % "3.3.5" % "provided",
  "org.apache.hadoop" % "hadoop-aws" % "3.3.5" % "provided",
  "com.amazonaws" % "aws-java-sdk-bundle" % "1.12.454" % "provided"
)
