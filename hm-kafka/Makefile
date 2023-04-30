# Kubernetes
kubernetes-kafka-topic-list:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-topics.sh \
			--bootstrap-server=localhost:9092 \
			--list
kubernetes-kafka-topic-delete:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-topics.sh \
			--bootstrap-server=localhost:9092 \
			--delete \
			--topic=my-topic
kubernetes-kafka-topic-describe:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-topics.sh \
			--bootstrap-server=localhost:9092 \
			--describe \
			--topic=my-topic
kubernetes-kafka-topic-message-count:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-run-class.sh \
			kafka.tools.GetOffsetShell \
			--bootstrap-server=localhost:9092 \
			--topic=my-topic \
			--time=-1

kubernetes-kafka-console-producer:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-producer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=my-topic
kubernetes-kafka-console-consumer:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=my-topic \
			--property=print.key=true
kubernetes-kafka-console-consumer-from-beginning:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=my-topic \
			--property=print.key=true \
			--from-beginning

kubernetes-kafka-avro-console-producer:
	kubectl exec confluent-schema-registry-0 --namespace=hm-confluent-schema-registry --stdin --tty -- \
		kafka-avro-console-producer \
            --bootstrap-server=hm-kafka-kafka-bootstrap.hm-kafka.svc:9092 \
            --topic=my-avro-topic \
            --property=schema.registry.url=http://localhost:8081 \
            --property=value.schema='{"type": "record", "name": "myrecord", "fields": [{"name": "f1", "type": "string"}]}'
	# {"f1": "hi"}
kubernetes-kafka-avro-console-consumer:
	kubectl exec confluent-schema-registry-0 --namespace=hm-confluent-schema-registry --stdin --tty -- \
		kafka-avro-console-consumer \
            --bootstrap-server=hm-kafka-kafka-bootstrap.hm-kafka.svc:9092 \
            --topic=my-avro-topic \
            --property=schema.registry.url=http://localhost:8081
kubernetes-kafka-avro-console-consumer-from-beginning:
	kubectl exec confluent-schema-registry-0 --namespace=hm-confluent-schema-registry --stdin --tty -- \
		kafka-avro-console-consumer \
            --bootstrap-server=hm-kafka-kafka-bootstrap.hm-kafka.svc:9092 \
            --topic=my-avro-topic \
            --property=schema.registry.url=http://localhost:8081 \
            --from-beginning

keytool-list-hm-kafka-kafka-truststore:
	cd .. && \
	keytool \
		-list \
		-keystore kubernetes/data/hm-kafka/hm-kafka/kafka-truststore.jks \
		-storepass m1Uaf4Crxzoo2Zxp
keytool-list-opa-kafka-connect-elasticsearch-keystore:
	cd .. && \
	keytool \
		-list \
		-keystore kubernetes/data/hm-kafka/opa-kafka-connect/elasticsearch-keystore.jks \
		-storepass MPx57vkACsRWKVap
keytool-list-logging-kafka-connect-elasticsearch-keystore:
	cd .. && \
	keytool \
		-list \
		-keystore kubernetes/data/hm-kafka/logging-kafka-connect/elasticsearch-keystore.jks \
		-storepass MPx57vkACsRWKVap

# Local
kafka-install:
	brew install kafka

zookeeper-start:
	zookeeper-server-start /opt/homebrew/etc/kafka/zookeeper.properties
kafka-start:
	kafka-server-start /opt/homebrew/etc/kafka/server.properties

zookeeper-status:
	zkServer status

kafka-topic-create:
	kafka-topics --bootstrap-server=localhost:9092 --create --topic=my-topic --partitions=3 --replication-factor=1
kafka-topic-delete:
	kafka-topics --bootstrap-server=localhost:9092 --delete --topic=my-topic
kafka-topic-list:
	kafka-topics --bootstrap-server=localhost:9092 --list
kafka-topic-describe:
	kafka-topics --bootstrap-server=localhost:9092 --describe --topic=my-topic

kafka-console-producer:
	kafka-console-producer --bootstrap-server=localhost:9092 --topic=my-topic
kafka-console-consumer:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --property=print.key=true
kafka-console-consumer-from-beginning:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --property=print.key=true --from-beginning
kafka-console-consumer-group:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --property=print.key=true --group=my-group

kafka-consumer-group-list:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --list
kafka-consumer-group-describe:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --describe --group=my-group
kafka-consumer-group-reset-offset-to-earliest:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --group=my-group --topic=my-topic --reset-offsets --to-earliest --execute
kafka-consumer-group-reset-offset-shift-by:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --group=my-group --topic=my-topic --reset-offsets --shift-by=-1 --execute
