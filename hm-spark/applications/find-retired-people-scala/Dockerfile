FROM docker.io/sbtscala/scala-sbt:eclipse-temurin-jammy-17.0.5_8_1.8.2_2.12.17 AS builder
WORKDIR /usr/src/app/
COPY hm-spark/applications/find-retired-people-scala/src/ ./src/
COPY hm-spark/applications/find-retired-people-scala/build.sbt ./
RUN sbt package

FROM docker.io/apache/spark:v3.3.2
WORKDIR /opt/spark/work-dir
COPY --from=builder /usr/src/app/target/scala-2.12/findretiredpeople_2.12-1.0.jar /opt/spark/work-dir/
