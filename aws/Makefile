# Amazon EMR
emr-cluster-create:
	aws emr create-cluster xxx
emr-cluster-list:
	aws emr list-clusters
emr-cluster-describe:
	aws emr describe-cluster --cluster-id=xxx
emr-cluster-terminate:
	aws emr terminate-clusters --cluster-ids=xxx

# Amazon MSK
msk-cluster-list:
	aws kafka list-clusters
msk-cluster-describe:
	aws kafka describe-cluster --cluster-arn=arn:aws:kafka:us-west-2:xxx

# Amazon MSK | EC2
msk-ec2-kafka-topic-list:
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=kafka-client/client.properties \
		--list
msk-ec2-kafka-topic-list-debug:
	KAFKA_OPTS=-Dlog4j.configuration=file:config/log4j.properties \
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=kafka-client.properties \
		--list
