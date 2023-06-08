#!/usr/bin/env bash
set -e

wget --no-verbose https://d1i4a15mxbxib1.cloudfront.net/api/plugins/confluentinc/kafka-connect-avro-converter/versions/7.4.0/confluentinc-kafka-connect-avro-converter-7.4.0.zip
wget --no-verbose https://d1i4a15mxbxib1.cloudfront.net/api/plugins/confluentinc/kafka-connect-jdbc/versions/10.7.2/confluentinc-kafka-connect-jdbc-10.7.2.zip
# Unable to use a version >= 19.0 due to https://stackoverflow.com/a/40691831/2000548
wget --no-verbose https://repo1.maven.org/maven2/com/google/guava/guava/18.0/guava-18.0.jar

unzip confluentinc-kafka-connect-avro-converter-7.4.0.zip
unzip confluentinc-kafka-connect-jdbc-10.7.2.zip

mkdir jdbc-sink-plugin

cp -R confluentinc-kafka-connect-avro-converter-7.4.0/lib/. jdbc-sink-plugin
cp -R confluentinc-kafka-connect-jdbc-10.7.2/lib/. jdbc-sink-plugin
cp guava-18.0.jar jdbc-sink-plugin

zip -r jdbc-sink-plugin.zip jdbc-sink-plugin

rm -f confluentinc-kafka-connect-avro-converter-7.4.0.zip
rm -f confluentinc-kafka-connect-jdbc-10.7.2.zip
rm -f guava-18.0.jar
rm -rf confluentinc-kafka-connect-avro-converter-7.4.0
rm -rf confluentinc-kafka-connect-jdbc-10.7.2
rm -rf jdbc-sink-plugin
