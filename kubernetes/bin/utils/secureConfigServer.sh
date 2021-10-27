#!/usr/bin/env bash

set -e


config_server_data_path="kubernetes/data/config-server"


echo "# Clean config server cert and key"
rm -f "${config_server_data_path}/server.crt"
rm -f "${config_server_data_path}/server.key"

echo "# Generate config server cert and key"
openssl req -x509 -newkey=rsa:4096 -nodes -out="${config_server_data_path}/server.crt" -keyout="${config_server_data_path}/server.key" -days=3650 -subj="/C=US/ST=CA/CN=hongbomiao.com"
echo "=================================================="
