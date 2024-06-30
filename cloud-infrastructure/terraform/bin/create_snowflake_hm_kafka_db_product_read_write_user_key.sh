#!/usr/bin/env bash
set -e

cd ~/.ssh

# Production
openssl genrsa 4096 | openssl pkcs8 -topk8 -inform PEM -out snowflake_production_hm_kafka_db_product_read_write_user_key.p8
# Private key passphrase: xxx
openssl rsa -in snowflake_production_hm_kafka_db_product_read_write_user_key.p8 -pubout -out snowflake_production_hm_kafka_db_product_read_write_user_key.pub
