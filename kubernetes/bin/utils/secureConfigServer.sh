#!/usr/bin/env bash

set -e


CONFIG_SERVER_DATA_PATH="kubernetes/data/config-server"


echo "# Clean config server cert and key"
rm -f "${CONFIG_SERVER_DATA_PATH}/server.crt"
rm -f "${CONFIG_SERVER_DATA_PATH}/server.key"

echo "# Generate config server cert and key"
openssl req -x509 -newkey=rsa:4096 -nodes -out="${CONFIG_SERVER_DATA_PATH}/server.crt" -keyout="${CONFIG_SERVER_DATA_PATH}/server.key" -days=3650 -subj="/C=US/ST=CA/CN=hongbomiao.com"
echo "=================================================="
