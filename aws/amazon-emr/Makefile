emr-cluster-create:
	aws emr create-cluster xxx
emr-cluster-list:
	aws emr list-clusters
emr-cluster-describe:
	aws emr describe-cluster --cluster-id=xxx
emr-cluster-terminate:
	aws emr terminate-clusters --cluster-ids=xxx

emr-hadoop-yarn-status:
	sudo systemctl status hadoop-yarn-resourcemanager
emr-hadoop-yarn-start:
	sudo systemctl start hadoop-yarn-resourcemanager
emr-hadoop-yarn-stop:
	sudo systemctl stop hadoop-yarn-resourcemanager

upload-bootstrap-actions:
	aws s3 cp bootstrap-actions/set_up.sh s3://hongbomiao-bucket/amazon-emr/bootstrap-actions/set_up.sh
