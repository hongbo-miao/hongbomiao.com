name := "Hello World"

version := "0.1"

scalaVersion := "2.13.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0",
  "org.apache.spark" %% "spark-sql" % "3.3.0",
  "org.apache.spark" %% "spark-streaming" % "3.3.0",
)
