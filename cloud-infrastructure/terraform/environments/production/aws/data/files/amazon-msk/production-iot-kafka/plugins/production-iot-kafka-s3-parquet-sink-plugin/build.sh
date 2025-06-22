#!/usr/bin/env bash
set -e

# Input
eval "$(jq --raw-output '@sh "KAFKA_PLUGIN_NAME=\(.kafka_plugin_name) CONFLUENT_KAFKA_CONNECT_S3_CONVERTER_VERSION=\(.confluent_kafka_connect_s3_converter_version) CONFLUENT_KAFKA_CONNECT_AVRO_CONVERTER_VERSION=\(.confluent_kafka_connect_avro_converter_version) LOCAL_DIR_PATH=\(.local_dir_path) LOCAL_FILE_NAME=\(.local_file_name)"')"

# Prepare
mkdir -p "/tmp/$KAFKA_PLUGIN_NAME/raw/"
mkdir -p "/tmp/$KAFKA_PLUGIN_NAME/temp/"
mkdir -p "/tmp/$KAFKA_PLUGIN_NAME/output/"

# Download
curl --silent --fail --show-error --location --output="/tmp/$KAFKA_PLUGIN_NAME/raw/confluentinc-kafka-connect-s3-$CONFLUENT_KAFKA_CONNECT_S3_CONVERTER_VERSION.zip" "https://d2p6pa21dvn84.cloudfront.net/api/plugins/confluentinc/kafka-connect-s3/versions/$CONFLUENT_KAFKA_CONNECT_S3_CONVERTER_VERSION/confluentinc-kafka-connect-s3-$CONFLUENT_KAFKA_CONNECT_S3_CONVERTER_VERSION.zip"
curl --silent --fail --show-error --location --output="/tmp/$KAFKA_PLUGIN_NAME/raw/confluentinc-kafka-connect-avro-converter-$CONFLUENT_KAFKA_CONNECT_AVRO_CONVERTER_VERSION.zip" "https://d2p6pa21dvn84.cloudfront.net/api/plugins/confluentinc/kafka-connect-avro-converter/versions/$CONFLUENT_KAFKA_CONNECT_AVRO_CONVERTER_VERSION/confluentinc-kafka-connect-avro-converter-$CONFLUENT_KAFKA_CONNECT_AVRO_CONVERTER_VERSION.zip"

# Build
unzip -q "/tmp/$KAFKA_PLUGIN_NAME/raw/confluentinc-kafka-connect-s3-$CONFLUENT_KAFKA_CONNECT_S3_CONVERTER_VERSION.zip" -d "/tmp/$KAFKA_PLUGIN_NAME/temp/"
cp -R "/tmp/$KAFKA_PLUGIN_NAME/temp/confluentinc-kafka-connect-s3-$CONFLUENT_KAFKA_CONNECT_S3_CONVERTER_VERSION/lib/." "/tmp/$KAFKA_PLUGIN_NAME/output/"
unzip -q "/tmp/$KAFKA_PLUGIN_NAME/raw/confluentinc-kafka-connect-avro-converter-$CONFLUENT_KAFKA_CONNECT_AVRO_CONVERTER_VERSION.zip" -d "/tmp/$KAFKA_PLUGIN_NAME/temp/"
cp -R "/tmp/$KAFKA_PLUGIN_NAME/temp/confluentinc-kafka-connect-avro-converter-$CONFLUENT_KAFKA_CONNECT_AVRO_CONVERTER_VERSION/lib/." "/tmp/$KAFKA_PLUGIN_NAME/output/"
rm -f "$LOCAL_DIR_PATH/$LOCAL_FILE_NAME"
zip --quiet --recurse-paths --junk-paths "$LOCAL_DIR_PATH/$LOCAL_FILE_NAME" "/tmp/$KAFKA_PLUGIN_NAME/output/"

# Clean
rm -r -f "/tmp/$KAFKA_PLUGIN_NAME/"

# Output
echo "{\"local_file_path\":\"$LOCAL_DIR_PATH/$LOCAL_FILE_NAME\"}"
