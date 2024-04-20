#!/usr/bin/env bash

cd ~/.ssh || exit

# Development
openssl genrsa 4096 | openssl pkcs8 -topk8 -inform PEM -out snowflake_development_hm_terraform_read_write_user_key.p8
# Private key passphrase: xxx
openssl rsa -in snowflake_development_hm_terraform_read_write_user_key.p8 -pubout -out snowflake_development_hm_terraform_read_write_user_key.pub

# Production
openssl genrsa 4096 | openssl pkcs8 -topk8 -inform PEM -out snowflake_production_hm_terraform_read_write_user_key.p8
# Private key passphrase: xxx
openssl rsa -in snowflake_production_hm_terraform_read_write_user_key.p8 -pubout -out snowflake_production_hm_terraform_read_write_user_key.pub
