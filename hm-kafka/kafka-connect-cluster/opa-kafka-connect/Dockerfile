FROM docker.io/alpine:3.21.1 AS builder
USER root:root
RUN mkdir -p /opt/kafka/plugins/ \
  # debezium-connector-postgres
  # https://mvnrepository.com/artifact/io.debezium/debezium-connector-postgres \
  && wget --no-verbose --output-document=debezium-connector-postgres.tar.gz https://repo1.maven.org/maven2/io/debezium/debezium-connector-postgres/2.7.1.Final/debezium-connector-postgres-2.7.1.Final-plugin.tar.gz \
  && tar --extract --file=debezium-connector-postgres.tar.gz -C /opt/kafka/plugins/ \
  && rm -f debezium-connector-postgres.tar.gz \
  # confluentinc-kafka-connect-elasticsearch
  # https://www.confluent.io/hub/confluentinc/kafka-connect-elasticsearch
  && wget --no-verbose --output-document=confluentinc-kafka-connect-elasticsearch.zip https://d2p6pa21dvn84.cloudfront.net/api/plugins/confluentinc/kafka-connect-elasticsearch/versions/14.1.1/confluentinc-kafka-connect-elasticsearch-14.1.1.zip \
  && mkdir -p /opt/kafka/plugins/confluentinc-kafka-connect-elasticsearch/ \
  && unzip confluentinc-kafka-connect-elasticsearch.zip -d /opt/kafka/plugins/confluentinc-kafka-connect-elasticsearch/ \
  && rm -f confluentinc-kafka-connect-elasticsearch.zip \
  # http-connector-for-apache-kafka
  # https://github.com/aiven/http-connector-for-apache-kafka
  && wget --no-verbose --output-document=http-connector-for-apache-kafka.zip https://github.com/Aiven-Open/http-connector-for-apache-kafka/archive/refs/tags/v0.8.0.zip \
  && mkdir -p /opt/kafka/plugins/http-connector-for-apache-kafka/ \
  && unzip http-connector-for-apache-kafka.zip -d /opt/kafka/plugins/http-connector-for-apache-kafka/ \
  && rm -f http-connector-for-apache-kafka.zip
USER 1001

FROM quay.io/strimzi/kafka:0.37.0-kafka-3.4.0
COPY --from=builder /opt/kafka/plugins/ /opt/kafka/plugins/
