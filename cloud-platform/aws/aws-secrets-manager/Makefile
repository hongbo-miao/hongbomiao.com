aws-secrets-manager-secret-list:
	aws secretsmanager list-secrets

# RDS - hm_iot_db
aws-secrets-manager-secret-create-hm-iot-db:
	aws secretsmanager create-secret \
        --name=hm-iot-rds/hm_iot_db/public/readonly \
        --description="hm-iot-rds/hm_iot_db/public/readonly credentials" \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-iot-rds-secret.json
aws-secrets-manager-secret-delete-hm-iot-db:
	aws secretsmanager delete-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:hm-iot-rds/hm_iot_db/public/readonly-xxxxxx \
		--force-delete-without-recovery
aws-secrets-manager-secret-describe-hm-iot-db:
	aws secretsmanager describe-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:hm-iot-rds/hm_iot_db/public/readonly-xxxxxx

# Snowflake - production_tracker_db
aws-secrets-manager-secret-create-production-tracker-db:
	aws secretsmanager create-secret \
        --name=hm-snowflake/production_tracker_db/public/readwrite \
        --description="hm-snowflake/production_tracker_db/public/readwrite credentials" \
        --region=us-west-2 \
        --secret-string=file://secrets/tracker-snowflake-credentials.json
aws-secrets-manager-secret-delete-production-tracker-db:
	aws secretsmanager delete-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:hm-snowflake/production_tracker_db/public/readwrite-xxxxxx \
		--force-delete-without-recovery
aws-secrets-manager-secret-describe-hm-iot-db:
	aws secretsmanager describe-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:hm-snowflake/production_tracker_db/public/readwrite-xxxxxx \
