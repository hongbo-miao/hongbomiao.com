FROM docker.io/sbtscala/scala-sbt:eclipse-temurin-jammy-17.0.5_8_1.9.0_2.12.17 AS builder
WORKDIR /usr/src/app/
COPY data-processing/hm-spark/applications/ingest-from-s3-to-kafka/.jvmopts ./
COPY data-processing/hm-spark/applications/ingest-from-s3-to-kafka/build.sbt ./
COPY data-processing/hm-spark/applications/ingest-from-s3-to-kafka/project/plugins.sbt ./project/
COPY data-processing/hm-spark/applications/ingest-from-s3-to-kafka/src/ ./src/
RUN sbt compile assembly

FROM docker.io/bitnami/spark:3.5.4-debian-12-r3
WORKDIR /opt/spark/work-dir
COPY --from=builder /usr/src/app/target/scala-2.12/IngestFromS3ToKafka-assembly-1.0.jar /opt/spark/work-dir/
