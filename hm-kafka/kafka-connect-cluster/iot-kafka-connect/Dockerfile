FROM docker.io/alpine:3.21.1 AS builder
USER root:root
RUN mkdir -p /opt/kafka/plugins/ \
  # apicurio-registry-distro-connect-converter
  # https://mvnrepository.com/artifact/io.apicurio/apicurio-registry-distro-connect-converter
  # && wget --no-verbose --output-document=apicurio-registry-distro-connect-converter.tar.gz https://repo1.maven.org/maven2/io/apicurio/apicurio-registry-distro-connect-converter/3.0.0.M3/apicurio-registry-distro-connect-converter-3.0.0.M3.tar.gz \
  # && mkdir -p /opt/kafka/plugins/apicurio-registry-distro-connect-converter/ \
  # && tar -x -f apicurio-registry-distro-connect-converter.tar.gz -C /opt/kafka/plugins/apicurio-registry-distro-connect-converter/ \
  # && rm -f apicurio-registry-distro-connect-converter.tar.gz \
  # kafka-connect-avro-converter
  # https://www.confluent.io/hub/confluentinc/kafka-connect-avro-converter
   && wget --no-verbose --output-document=kafka-connect-avro-converter.zip https://d2p6pa21dvn84.cloudfront.net/api/plugins/confluentinc/kafka-connect-avro-converter/versions/7.7.0/confluentinc-kafka-connect-avro-converter-7.7.0.zip \
   && mkdir -p /opt/kafka/plugins/kafka-connect-avro-converter/ \
   && unzip kafka-connect-avro-converter.zip -d /opt/kafka/plugins/kafka-connect-avro-converter/ \
   && rm -f kafka-connect-avro-converter.zip \
  # confluentinc-kafka-connect-jdbc
  # https://www.confluent.io/hub/confluentinc/kafka-connect-jdbc
  && wget --no-verbose --output-document=confluentinc-kafka-connect-jdbc.zip https://d2p6pa21dvn84.cloudfront.net/api/plugins/confluentinc/kafka-connect-jdbc/versions/10.7.6/confluentinc-kafka-connect-jdbc-10.7.6.zip \
  && mkdir -p /opt/kafka/plugins/confluentinc-kafka-connect-jdbc/ \
  && unzip confluentinc-kafka-connect-jdbc.zip -d /opt/kafka/plugins/confluentinc-kafka-connect-jdbc/ \
  && rm -f confluentinc-kafka-connect-jdbc.zip
USER 1001

FROM quay.io/strimzi/kafka:0.37.0-kafka-3.4.0
USER root:root
COPY --from=builder /opt/kafka/plugins/ /opt/kafka/plugins/
USER 1001
