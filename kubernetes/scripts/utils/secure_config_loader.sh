#!/usr/bin/env bash
set -e

CONFIG_LOADER_DATA_PATH="kubernetes/data/config-loader"

echo "# Clean config loader cert and key"
rm -f "${CONFIG_LOADER_DATA_PATH}/server.crt"
rm -f "${CONFIG_LOADER_DATA_PATH}/server.key"
echo "=================================================="

echo "# Generate config loader cert and key"
openssl req -x509 -newkey=rsa:4096 -nodes -out="${CONFIG_LOADER_DATA_PATH}/server.crt" -keyout="${CONFIG_LOADER_DATA_PATH}/server.key" -days=3650 -subj="/C=US/ST=CA/CN=hongbomiao.com"
echo "=================================================="
