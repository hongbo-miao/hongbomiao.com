#!/usr/bin/env bash
set -e

# SSH keys
echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPWhR5NV13iw0X8lKqsrSRqbcIJcA5AVMjyfJjOrplwH hongbo-miao >> /home/hadoop/.ssh/authorized_keys

# Delta Lake
# https://github.com/aws-samples/emr-studio-notebook-examples/blob/main/examples/deltalake-example-notebook-pyspark.ipynb
# https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-6110-release.html
sudo curl --remote-name --output-dir /usr/lib/spark/jars/ https://repo1.maven.org/maven2/io/delta/delta-core_2.12/2.2.0/delta-core_2.12-2.2.0.jar
sudo curl --remote-name --output-dir /usr/lib/spark/jars/ https://repo1.maven.org/maven2/io/delta/delta-storage/2.2.0/delta-storage-2.2.0.jar
