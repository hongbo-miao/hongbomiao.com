#!/usr/bin/env bash
set -e

cd ~/.ssh

# production_engineer_db - motor - read_only_service_account
openssl genrsa 4096 | openssl pkcs8 -topk8 -inform PEM -out snowflake_production_engineer_db_motor_read_only_service_account_key.p8
# Private key passphrase: xxx
openssl rsa -in snowflake_production_engineer_db_motor_read_only_service_account_key.p8 -pubout -out snowflake_production_engineer_db_motor_read_only_service_account_key.pub

# production_engineer_db - motor - read_write_service_account
openssl genrsa 4096 | openssl pkcs8 -topk8 -inform PEM -out snowflake_production_engineer_db_motor_read_write_service_account_key.p8
# Private key passphrase: xxx
openssl rsa -in snowflake_production_engineer_db_motor_read_write_service_account_key.p8 -pubout -out snowflake_production_engineer_db_motor_read_write_service_account_key.pub
