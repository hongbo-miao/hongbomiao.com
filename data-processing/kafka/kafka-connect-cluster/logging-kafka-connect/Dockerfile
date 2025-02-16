FROM docker.io/alpine:3.21.3 AS builder
USER root:root
RUN mkdir -p /opt/kafka/plugins/ \
  # confluentinc-kafka-connect-elasticsearch
  # https://www.confluent.io/hub/confluentinc/kafka-connect-elasticsearch
  && wget --no-verbose --output-document=confluentinc-kafka-connect-elasticsearch.zip https://d2p6pa21dvn84.cloudfront.net/api/plugins/confluentinc/kafka-connect-elasticsearch/versions/14.1.1/confluentinc-kafka-connect-elasticsearch-14.1.1.zip \
  && mkdir -p /opt/kafka/plugins/confluentinc-kafka-connect-elasticsearch/ \
  && unzip confluentinc-kafka-connect-elasticsearch.zip -d /opt/kafka/plugins/confluentinc-kafka-connect-elasticsearch/ \
  && rm -f confluentinc-kafka-connect-elasticsearch.zip
USER 1001

FROM quay.io/strimzi/kafka:0.37.0-kafka-3.4.0
COPY --from=builder /opt/kafka/plugins/ /opt/kafka/plugins/
