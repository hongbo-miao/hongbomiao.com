# Kafka
amazon-msk-kafka-cluster-list:
	aws kafka list-clusters
amazon-msk-kafka-cluster-describe:
	aws kafka describe-cluster --cluster-arn=arn:aws:kafka:us-west-2:272394222652:cluster/hm-msk-cluster/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-xx

# EC2
kafka-topic-list:
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/iam-client.properties \
		--list
kafka-topic-list-debug:
	KAFKA_OPTS=-Dlog4j.configuration=file:config/log4j.properties \
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/iam-client.properties \
		--list
kafka-topic-create:
	KAFKA_HEAP_OPTS="-Xmx2g" \
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/iam-client.properties \
		--create \
		--topic=my-topic \
		--partitions=3 \
		--replication-factor=2
kafka-topic-create:
	bin/kafka-topics.sh \
		--bootstrap-server=xxx \
		--command-config=config/iam-client.properties \
		--delete \
		--topic=my-topic
kafka-topic-producer:
	bin/kafka-console-producer.sh \
		--bootstrap-server=xxx \
		--producer.config=config/iam-client.properties \
		--topic=my-topic
kafka-topic-consume:
	bin/kafka-console-consumer.sh \
		--bootstrap-server=xxx \
		--consumer.config=config/iam-client.properties \
		--topic=my-topic \
		--property=print.key=true
