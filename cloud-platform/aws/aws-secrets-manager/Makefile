aws-secrets-manager-secret-create:
	aws secretsmanager create-secret \
        --name=hm-iot-rds/hm_iot_db/readonly \
        --description="RDS credentials for hm_iot_db" \
        --region=us-west-2 \
        --secret-string=file://secrets/hm-iot-rds-secret.json
aws-secrets-manager-secret-delete:
	aws secretsmanager delete-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:hm-iot-rds/hm_iot_db/readwrite-xxxxxx \
		--force-delete-without-recovery
aws-secrets-manager-secret-describe:
	aws secretsmanager describe-secret \
		--secret-id=arn:aws:secretsmanager:us-west-2:272394222652:secret:hm-iot-rds/hm_iot_db/readwrite-xxxxxx
aws-secrets-manager-secret-list:
	aws secretsmanager list-secrets
