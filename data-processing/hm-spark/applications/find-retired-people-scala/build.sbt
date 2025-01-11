name := "FindRetiredPeople"
version := "1.0"
scalaVersion := "2.13.16"

val sparkVersion = "3.5.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
)

// Scalafix
semanticdbEnabled := true
semanticdbVersion := scalafixSemanticdb.revision // Required for Scala 2.x
scalacOptions ++= Seq(
  "-Ywarn-unused-import" // Required by RemoveUnused rule
)
