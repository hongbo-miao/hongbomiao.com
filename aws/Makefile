aws-profile-list:
	aws configure list-profiles
aws-profile-show:
	aws configure list --profile=default

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
kafka-topic-list:
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/client.properties \
		--list
kafka-topic-list-debug:
	KAFKA_OPTS=-Dlog4j.configuration=file:config/log4j.properties \
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/client.properties \
		--list
kafka-topic-create:
	KAFKA_HEAP_OPTS="-Xmx2g" \
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/client.properties \
		--create \
		--topic=my-topic \
		--partitions=3 \
		--replication-factor=2
kafka-topic-create:
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/client.properties \
		--delete \
		--topic=my-topic
kafka-topic-producer:
	bin/kafka-console-producer.sh \
		--bootstrap-server=xxx \
		--producer.config=config/client.properties \
		--topic=my-topic
kafka-topic-consume:
	bin/kafka-console-consumer.sh \
		--bootstrap-server=xxx \
		--consumer.config=config/client.properties \
		--topic=my-topic \
		--property=print.key=true
