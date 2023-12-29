#!/usr/bin/env bash
set -e

echo "# Add SSH keys"
echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPWhR5NV13iw0X8lKqsrSRqbcIJcA5AVMjyfJjOrplwH hongbo-miao >> /home/hadoop/.ssh/authorized_keys

echo "# Install Python"
# https://github.com/aws-samples/aws-emr-utilities/blob/main/utilities/emr-ec2-custom-python3/README.md
# Update the corresponding hm_sedona_emr -> PYSPARK_PYTHON version in terraform/main.tf
PYTHON_VERSION=3.11.7
sudo yum --assumeyes remove openssl-devel*
sudo yum --assumeyes install \
  bzip2-devel \
  expat-devel \
  gcc \
  libffi-devel \
  make \
  openssl11-devel \
  tar
curl --silent --fail --show-error --location "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" | tar -x -J -v
cd "Python-${PYTHON_VERSION}"
export CFLAGS="-march=native"
./configure \
  --enable-loadable-sqlite-extensions \
  --with-dtrace \
  --with-lto \
  --enable-optimizations \
  --with-system-expat \
  --prefix="/usr/local/python${PYTHON_VERSION}"
sudo make altinstall
sudo "/usr/local/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION%.*}" -m pip install --upgrade pip

echo "# Install dependencies"
sudo curl --silent --fail --show-error --location --remote-name --output-dir /usr/lib/spark/jars/ https://repo1.maven.org/maven2/org/apache/sedona/sedona-spark-shaded-3.4_2.12/1.5.0/sedona-spark-shaded-3.4_2.12-1.5.0.jar
sudo curl --silent --fail --show-error --location --remote-name --output-dir /usr/lib/spark/jars/ https://repo1.maven.org/maven2/org/datasyslab/geotools-wrapper/1.5.0-28.2/geotools-wrapper-1.5.0-28.2.jar
"/usr/local/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION%.*}" -m pip install \
  apache-sedona[spark]==1.5.0 \
  attrs==23.1.0 \
  descartes==1.1.0 \
  geopandas==0.14.1 \
  matplotlib==3.8.2 \
  pandas==2.1.4 \
  shapely==2.0.2
