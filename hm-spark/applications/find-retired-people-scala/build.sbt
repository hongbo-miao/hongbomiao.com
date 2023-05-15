name := "FindRetiredPeople"
version := "1.0"
scalaVersion := "2.12.17"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.2" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.3.2" % "provided"
)

// Scalafix
semanticdbEnabled := true
semanticdbVersion := scalafixSemanticdb.revision // Required for Scala 2.x
scalacOptions ++= Seq(
  "-Ywarn-unused-import" // Required by RemoveUnused rule
)
