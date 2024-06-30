aws-secrets-manager-secret-list:
	aws secretsmanager list-secrets

aws-secrets-manager-secret-describe:
	aws secretsmanager describe-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:xxx
aws-secrets-manager-secret-delete:
	aws secretsmanager delete-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:xxx \
		--force-delete-without-recovery

aws-secrets-manager-secret-create-hm-amazon-rds-production-hm-iot-db-public-read-only:
	aws secretsmanager create-secret \
        --name=hm/amazon-rds/production_hm_iot_db/public/read_only \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-amazon-rds-production-hm-iot-db-public-read-only-user-credentials.json

# Snowflake - HM_KAFKA_DB database PRODUCT schema
aws-secrets-manager-secret-create-hm-snowflake-production-hm-kafka-db-product-read-write:
	aws secretsmanager create-secret \
        --name=hm/snowflake/production_hm_kafka_db/product/read_write \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-snowflake-hm-production-hm-kafka-db-product-read-write-user-credentials.json

# Airbyte - Amazon RDS
aws-secrets-manager-secret-create-production-hm-airbyte-postgres:
	aws secretsmanager create-secret \
        --name=production-hm-airbyte-postgres/admin \
        --region=us-west-2 \
        --secret-string=file://secrets/production-hm-airbyte-postgres-admin-credentials.json
