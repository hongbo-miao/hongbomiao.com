amazon-emr-cluster-create:
	aws emr create-cluster \
		--name=hm-amazon-emr-cluster-trino \
		--release-label=emr-6.12.0 \
		--applications=Name=Trino \
		--configurations=file://hm-amazon-emr-cluster-trino-config.json \
		--instance-groups=file://cluster-config/instance-groups.json \
		--bootstrap-actions='[{"Name": "set_up", "Path": "s3://hongbomiao-bucket/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"}]' \
		--log-uri=s3n://hongbomiao-bucket/amazon-emr/logs/ \
		--service-role=arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-xxxxxxxxxxxxxxx \
		--scale-down-behavior=TERMINATE_AT_TASK_COMPLETION \
		--auto-termination-policy='{"IdleTimeout":7200}' \
		--step-concurrency-level=10 \
		--region=us-west-2
amazon-s3-upload-bootstrap-actions:
	aws s3 cp bootstrap-actions/set_up.sh s3://hongbomiao-bucket/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh

# Amazon EMR - Trino
trino-status:
	sudo systemctl status trino-server
trino-start:
	sudo systemctl start trino-server
trino-stop:
	sudo systemctl stop trino-server
trino-show-config:
	sudo cat /etc/trino/conf/config.properties
trino-show-config-jvm:
	cat /etc/trino/conf/jvm.config
trino-check-health:
	curl localhost:8889/v1/status
trino-execute:
	trino-cli \
		--server=localhost:8889 \
		--execute='select * from delta.hm_iot_delta_db.motor;'
	trino-cli \
		--server=localhost:8889 \
		--execute='show catalogs;'
	trino-cli \
		--server=localhost:8889 \
		--catalog=delta \
		--execute='show schemas;'
	trino-cli \
		--server=localhost:8889 \
		--catalog=delta \
		--schema=hm_iot_delta_db \
		--execute='show tables;'

# Amazon EMR - Presto
presto-status:
	sudo systemctl status presto-server
presto-start:
	sudo systemctl start presto-server
presto-stop:
	sudo systemctl stop presto-server
presto-show-config:
	sudo cat /etc/presto/conf/config.properties
presto-execute:
	presto-cli \
		--server=localhost:8889 \
		--execute='show catalogs;'
	presto-cli \
		--server=localhost:8889 \
		--catalog=system \
		--execute='show schemas;'
	presto-cli \
		--server=localhost:8889 \
		--catalog=awsdatacatalog \
		--execute='show schemas;'
	presto-cli \
		--server=localhost:8889 \
		--catalog=awsdatacatalog \
		--schema=information_schema \
		--execute='show tables;'

# Amazon EMR - Hive
hive:
	hive
hive-show-hive-site:
	cat /etc/hive/conf/hive-site.xml
