#!/usr/bin/env bash
set -e

echo "# Add SSH keys"
# An empty line is necessary since the original public key lacks a newline character at the end
{
  echo
  echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPWhR5NV13iw0X8lKqsrSRqbcIJcA5AVMjyfJjOrplwH hongbo-miao"
} >> /home/hadoop/.ssh/authorized_keys

echo "# Install Amazon EMR cluster-scoped dependencies"
# https://sedona.apache.org/latest/setup/emr/
sudo curl --silent --fail --show-error --location --remote-name --output-dir /usr/lib/spark/jars/ https://repo1.maven.org/maven2/org/apache/sedona/sedona-spark-shaded-3.5_2.12/1.5.1/sedona-spark-shaded-3.5_2.12-1.5.1.jar
sudo curl --silent --fail --show-error --location --remote-name --output-dir /usr/lib/spark/jars/ https://repo1.maven.org/maven2/org/datasyslab/geotools-wrapper/1.5.1-28.2/geotools-wrapper-1.5.1-28.2.jar
sudo python3 -m pip install \
  apache-sedona==1.5.1 \
  h3==3.7.7
