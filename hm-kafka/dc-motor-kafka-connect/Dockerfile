FROM docker.io/alpine:3.17.3 AS builder
USER root:root
RUN mkdir -p /opt/kafka/plugins/ \
  # jdbc-connector-for-apache-kafka
  # https://github.com/aiven/jdbc-connector-for-apache-kafka
  && wget --no-verbose --output-document=jdbc-connector-for-apache-kafka.zip https://github.com/aiven/jdbc-connector-for-apache-kafka/releases/download/v6.8.0/jdbc-connector-for-apache-kafka-6.8.0.zip \
  && unzip jdbc-connector-for-apache-kafka.zip -d /opt/kafka/plugins/ \
  && rm -f jdbc-connector-for-apache-kafka.zip
USER 1001

FROM quay.io/strimzi/kafka:0.34.0-kafka-3.4.0
COPY --from=builder /opt/kafka/plugins/ /opt/kafka/plugins/
