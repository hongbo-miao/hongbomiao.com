FROM docker.io/alpine:3.17.1 AS builder
USER root:root
RUN mkdir -p /opt/kafka/plugins/ \
  # debezium-connector-postgres
  # https://mvnrepository.com/artifact/io.debezium/debezium-connector-postgres \
  && wget --no-verbose --output-document=debezium-connector-postgres.tar.gz https://repo1.maven.org/maven2/io/debezium/debezium-connector-postgres/1.7.1.Final/debezium-connector-postgres-1.7.1.Final-plugin.tar.gz \
  && tar -x -f debezium-connector-postgres.tar.gz -C /opt/kafka/plugins/ \
  && rm -f debezium-connector-postgres.tar.gz \
  # kafka-connect-elasticsearch
  # https://www.confluent.io/hub/confluentinc/kafka-connect-elasticsearch
  && wget --no-verbose --output-document=kafka-connect-elasticsearch.zip https://d1i4a15mxbxib1.cloudfront.net/api/plugins/confluentinc/kafka-connect-elasticsearch/versions/11.1.6/confluentinc-kafka-connect-elasticsearch-11.1.6.zip \
  && unzip kafka-connect-elasticsearch.zip -d /opt/kafka/plugins/ \
  && rm -f kafka-connect-elasticsearch.zip \
  # http-connector-for-apache-kafka
  # https://github.com/aiven/http-connector-for-apache-kafka
  && wget --no-verbose --output-document=http-connector-for-apache-kafka.zip https://github.com/aiven/http-connector-for-apache-kafka/releases/download/v0.5.0/http-connector-for-apache-kafka-0.5.0.zip \
  && unzip http-connector-for-apache-kafka.zip -d /opt/kafka/plugins/ \
  && rm -f http-connector-for-apache-kafka.zip
USER 1001


FROM quay.io/strimzi/kafka:0.29.0-kafka-3.0.0
COPY --from=builder /opt/kafka/plugins/ /opt/kafka/plugins/
