amazon-emr-cluster-create:
	aws emr create-cluster \
		--name=hm-amazon-emr-cluster-trino \
		--release-label=emr-6.13.0 \
		--applications=Name=Trino \
		--configurations=file://hm-amazon-emr-cluster-trino-config.json \
		--instance-fleets=file://cluster-config/instance-fleets.json \
		--bootstrap-actions='[{"Name": "set_up", "Path": "s3://hm-production-bucket/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"}]' \
		--log-uri=s3://hm-production-bucket/amazon-emr/logs/ \
		--service-role=arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-xxxxxxxxxxxxxxx \
		--scale-down-behavior=TERMINATE_AT_TASK_COMPLETION \
		--auto-termination-policy='{"IdleTimeout":7200}' \
		--step-concurrency-level=10 \
		--region=us-west-2
amazon-s3-upload-bootstrap-actions:
	aws s3 cp bootstrap-actions/set_up.sh s3://hm-production-bucket/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh

# Amazon EMR - Trino
trino-status:
	sudo systemctl status trino-server
trino-start:
	sudo systemctl start trino-server
trino-stop:
	sudo systemctl stop trino-server

trino-show-jvm-config:
	cat /etc/trino/conf/jvm.config
trino-show-trino-config:
	sudo cat /etc/trino/conf/config.properties
trino-show-trino-config-node:
	cat /etc/trino/conf/node.properties
trino-show-trino-config-catalog:
	cat /etc/trino/conf.dist/catalog/delta.properties
	cat /etc/trino/conf.dist/catalog/hive.properties
	cat /etc/trino/conf.dist/catalog/postgresql.properties

trino-web-ui:
	curl localhost:8889/ui/login.html
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

# Amazon EMR - Hive
hive:
	hive
hive-show-hive-site:
	cat /etc/hive/conf/hive-site.xml
