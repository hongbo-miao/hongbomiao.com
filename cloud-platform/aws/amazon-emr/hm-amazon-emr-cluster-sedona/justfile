amazon-emr-cluster-create:
	aws emr create-cluster \
		--name=hm-amazon-emr-cluster-jupterlab \
		--release-label=emr-6.13.0 \
		--applications Name=Hadoop Name=Hive Name=JupyterEnterpriseGateway Name=Spark \
		--instance-fleets=file://cluster-config/instance-fleets.json \
		--bootstrap-actions='[{"Name": "set_up", "Path": "s3://hm-production-bucket/amazon-emr/hm-amazon-emr-cluster-jupterlab/bootstrap-actions/set_up.sh"}]' \
		--log-uri=s3://hm-production-bucket/amazon-emr/logs/ \
		--service-role=arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-xxxxxxxxxxxxxxx \
		--scale-down-behavior=TERMINATE_AT_TASK_COMPLETION \
		--auto-termination-policy='{"IdleTimeout":7200}' \
		--step-concurrency-level=10 \
		--region=us-west-2
amazon-s3-upload-bootstrap-actions:
	aws s3 cp bootstrap-actions/set_up.sh s3://hm-production-bucket/amazon-emr/hm-amazon-emr-cluster-jupterlab/bootstrap-actions/set_up.sh

spark-show-spark-defaults:
	cat /etc/spark/conf/spark-defaults.conf

hadoop-show-yarn-site:
	cat /etc/hadoop/conf.empty/yarn-site.xml
