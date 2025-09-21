#!/usr/bin/env bash
set -e

cd ~/.ssh

# production_hm_airbyte_db - owner_service_account
openssl genrsa 4096 | openssl pkcs8 -topk8 -inform PEM -out snowflake_production_hm_airbyte_db_owner_service_account_key.p8
# Private key passphrase: xxx
openssl rsa -in snowflake_production_hm_airbyte_db_owner_service_account_key.p8 -pubout -out snowflake_production_hm_airbyte_db_owner_service_account_key.pub
