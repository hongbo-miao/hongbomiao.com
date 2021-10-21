FROM strimzi/kafka:0.20.1-kafka-2.6.0

USER root:root
RUN mkdir -p /opt/kafka/plugins/debezium

# Download, unpack, and place the debezium-connector-postgres folder into the /opt/kafka/plugins/debezium directory
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -s https://repo1.maven.org/maven2/io/debezium/debezium-connector-postgres/1.7.0.Final/debezium-connector-postgres-1.7.0.Final-plugin.tar.gz | tar xvz --transform 's/debezium-connector-postgres/debezium/' --directory /opt/kafka/plugins/

USER 1001
