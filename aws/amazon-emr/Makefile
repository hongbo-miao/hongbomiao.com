# Local
amazon-emr-cluster-create:
	aws emr create-cluster \
		--name=hm-amazon-emr-cluster \
		--applications=Name=Trino \
		--configurations=file://hm-amazon-emr-cluster-trino-config.json
amazon-emr-cluster-list:
	aws emr list-clusters
amazon-emr-cluster-describe:
	aws emr describe-cluster --cluster-id=xxx
amazon-emr-cluster-terminate:
	aws emr terminate-clusters --cluster-ids=xxx

amazon-s3-upload-bootstrap-actions:
	aws s3 cp bootstrap-actions/set_up.sh s3://hongbomiao-bucket/amazon-emr/bootstrap-actions/set_up.sh

# Amazon EMR - Hadoop
hadoop-yarn-status:
	sudo systemctl status hadoop-yarn-resourcemanager
hadoop-yarn-start:
	sudo systemctl start hadoop-yarn-resourcemanager
hadoop-yarn-stop:
	sudo systemctl stop hadoop-yarn-resourcemanager

# Amazon EMR - Trino
trino-status:
	sudo systemctl status trino-server
trino-start:
	sudo systemctl start trino-server
trino-stop:
	sudo systemctl stop trino-server
trino-show-config:
	sudo cat /etc/trino/conf/config.properties
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

# Amazon EMR - Hadoop HDFS
hadoop-hdfs-ls:
	hdfs dfs -ls
	hdfs dfs -ls /
	hdfs dfs -ls /user
	hdfs dfs -ls hdfs:///
	hdfs dfs -ls hdfs:///user
	hdfs dfs -ls s3://hongbomiao-bucket/
hadoop-hdfs-show-disk-free:
	hdfs dfs -df
hadoop-hdfs-show-disk-usage:
	hdfs dfs -du
hadoop-hdfs-copy-from-local:
	echo "Hello" >> file.txt
	hdfs dfs -copyFromLocal file.txt
hadoop-hdfs-cat:
	hdfs dfs -cat file.txt
hadoop-hdfs-mv:
	hdfs dfs -mv file.txt file2.txt
hadoop-hdfs-cp:
	hdfs dfs -cp file.txt file2.txt
hadoop-hdfs-rm:
	hdfs dfs -rm file
	hdfs dfs -rm -r folder
