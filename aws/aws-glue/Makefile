aws-glue-crawler-get-crawler:
	aws glue get-crawler --name=hm-delta-lake-crawler-iot

aws-glue-crawler-get-connection:
	aws glue get-connection --name=hm-postgres-connection

aws-glue-partition-index-list:
	aws glue get-partition-indexes --database-name=hm_delta_db --table-name=motor
aws-glue-partition-index-create:
	aws glue create-partition-index --database-name=hm_delta_db --table-name=motor --partition-index=Keys=_event_id,IndexName=_event_id_idx
aws-glue-partition-index-delete:
	aws glue delete-partition-index --database-name=hm_delta_db --table-name=motor --index-name=_event_id_idx

create-adsb-db-delta-tables:
	bash bin/create_adsb_db_delta_tables.sh
create-iot-db-delta-tables:
	bash bin/create_iot_db_delta_tables.sh
