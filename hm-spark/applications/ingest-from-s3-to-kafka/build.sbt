name := "IngestFromS3ToKafka"
version := "1.0"
scalaVersion := "2.12.18"
resolvers += "confluent" at "https://packages.confluent.io/maven/"

val sparkVersion = "3.5.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.hadoop" % "hadoop-common" % "3.3.6" % "provided",
  "org.apache.hadoop" % "hadoop-aws" % "3.3.6" % "provided",
  "com.amazonaws" % "aws-java-sdk-bundle" % "1.12.596" % "provided",

  "org.apache.spark" %% "spark-avro" % sparkVersion,
  "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion,
  "io.delta" %% "delta-core" % "2.4.0",
  "za.co.absa" %% "abris" % "6.3.0"
)

ThisBuild / assemblyMergeStrategy := {
  // https://stackoverflow.com/a/67937671/2000548
  case PathList("module-info.class") => MergeStrategy.discard
  case x if x.endsWith("/module-info.class") => MergeStrategy.discard
  // https://stackoverflow.com/a/76129963/2000548
  case PathList("org", "apache", "spark", "unused", "UnusedStubClass.class") => MergeStrategy.first
  case x =>
    val oldStrategy = (ThisBuild / assemblyMergeStrategy).value
    oldStrategy(x)
}

// Scalafix
semanticdbEnabled := true
semanticdbVersion := scalafixSemanticdb.revision // Required for Scala 2.x
scalacOptions ++= Seq(
  "-Ywarn-unused-import" // Required by RemoveUnused rule
)
