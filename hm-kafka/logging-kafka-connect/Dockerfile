FROM docker.io/alpine:3.17.3 AS builder
USER root:root
RUN mkdir -p /opt/kafka/plugins/ \
  # kafka-connect-elasticsearch
  # https://www.confluent.io/hub/confluentinc/kafka-connect-elasticsearch
  && wget --no-verbose --output-document=kafka-connect-elasticsearch.zip https://d1i4a15mxbxib1.cloudfront.net/api/plugins/confluentinc/kafka-connect-elasticsearch/versions/14.0.6/confluentinc-kafka-connect-elasticsearch-14.0.6.zip \
  && unzip kafka-connect-elasticsearch.zip -d /opt/kafka/plugins/ \
  && rm -f kafka-connect-elasticsearch.zip
USER 1001


FROM quay.io/strimzi/kafka:0.34.0-kafka-3.4.0
COPY --from=builder /opt/kafka/plugins/ /opt/kafka/plugins/
