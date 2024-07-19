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

# HM Jira
aws-secrets-manager-secret-create-hm-jira:
	aws secretsmanager create-secret \
        --name=hm-jira \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-jira-credentials.json

# Snowflake - HM_KAFKA_DB database PRODUCT schema
aws-secrets-manager-secret-create-hm-snowflake-production-hm-kafka-db-product-read-write:
	aws secretsmanager create-secret \
        --name=hm/snowflake/production_hm_kafka_db/product/read_write \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-snowflake-hm-production-hm-kafka-db-product-read-write-user-credentials.json

# Snowflake - HM_AIRBYTE_DB database
aws-secrets-manager-secret-create-snowflake-production-hm-airbyte-db-owner-user:
	aws secretsmanager create-secret \
        --name=hm/snowflake/production_hm_airbyte_db/owner \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-snowflake-hm-production-hm-airbyte-db-owner-user-credentials.json

# Airbyte - Amazon RDS
aws-secrets-manager-secret-create-production-hm-airbyte-postgres:
	aws secretsmanager create-secret \
        --name=production-hm-airbyte-postgres/admin \
        --region=us-west-2 \
        --secret-string=file://secrets/production-hm-airbyte-postgres-admin-credentials.json

# MLflow - Amazon RDS
aws-secrets-manager-secret-create-production-hm-mlflow-postgres:
	aws secretsmanager create-secret \
        --name=production-hm-mlflow-postgres/admin \
        --region=us-west-2 \
        --secret-string=file://secrets/production-hm-mlflow-postgres-admin-credentials.json

# Production IoT Kafka
# https://docs.aws.amazon.com/msk/latest/developerguide/msk-password.html
# - Secret name must start with "AmazonMSK_"
aws-secrets-manager-secret-create-snowflake-production-iot-kafka-producer:
	aws secretsmanager create-secret \
        --name=AmazonMSK_hm/production-iot-kafka/producer \
        --region=us-west-2 \
        --kms-key-id=arn:aws:kms:us-west-2:272394222652:key/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
        --secret-string=file://secrets/production-iot-kafka-credentials.json
