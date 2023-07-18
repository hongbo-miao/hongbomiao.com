# Local
emr-cluster-create:
	aws emr create-cluster xxx
emr-cluster-list:
	aws emr list-clusters
emr-cluster-describe:
	aws emr describe-cluster --cluster-id=xxx
emr-cluster-terminate:
	aws emr terminate-clusters --cluster-ids=xxx

s3-upload-bootstrap-actions:
	aws s3 cp bootstrap-actions/set_up.sh s3://hongbomiao-bucket/amazon-emr/bootstrap-actions/set_up.sh

# Amazon EMR - Hadoop
hadoop-yarn-status:
	sudo systemctl status hadoop-yarn-resourcemanager
hadoop-yarn-start:
	sudo systemctl start hadoop-yarn-resourcemanager
hadoop-yarn-stop:
	sudo systemctl stop hadoop-yarn-resourcemanager

# Amazon EMR - Trino
trino-show-config:
	sudo cat /etc/trino/conf/config.properties
trino-execute:
	trino-cli \
		--server=localhost:8889 \
		--execute='show catalogs;'
	trino-cli \
		--server=localhost:8889 \
		--catalog=hive \
		--execute='show schemas;'
	trino-cli \
		--server=localhost:8889 \
		--catalog=hudi \
		--execute='show schemas;'
	trino-cli \
		--server=localhost:8889 \
		--catalog=system \
		--execute='show schemas;'
	trino-cli \
		--server=localhost:8889 \
		--catalog=system \
		--schema=jdbc \
		--execute='show tables;'
