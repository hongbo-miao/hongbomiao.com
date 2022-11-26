# Docker
docker-build:
	cd .. && \
	docker build --file=kafka-connect/Dockerfile --tag=ghcr.io/hongbo-miao/hm-kafka-connect:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-kafka-connect:latest
docker-sh:
	docker run -it ghcr.io/hongbo-miao/hm-kafka-connect:latest sh

# Kubernetes
kafka-get-hm-kafkaconnect:
	kubectl get kafkaconnects hm-kafkaconnect --namespace=kafka --output=yaml
kafka-get-postgres-kafkaconnector:
	kubectl get kafkaconnectors postgres-kafkaconnector --namespace=kafka --output=yaml
kafka-get-elasticsearch-sink-kafkaconnector:
	kubectl get kafkaconnectors elasticsearch-sink-kafkaconnector --namespace=kafka --output=yaml
kafka-get-http-sink-kafkaconnector:
	kubectl get kafkaconnectors http-sink-kafkaconnector --namespace=kafka --output=yaml
kafka-list-topics:
	kubectl exec my-cluster-kafka-0 --namespace=kafka --container=kafka --stdin --tty -- \
		bin/kafka-topics.sh \
			--bootstrap-server=localhost:9092 \
			--list
kafka-consume:
	kubectl exec my-cluster-kafka-0 --namespace=kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=opa_db_server.public.role
kafka-consume-from-beginning:
	kubectl exec my-cluster-kafka-0 --namespace=kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=opa_db_server.public.role \
			--from-beginning

# Local
zookeeper-start:
	zookeeper-server-start /usr/local/etc/kafka/zookeeper.properties
kafka-start:
	kafka-server-start /usr/local/etc/kafka/server.properties

kafka-topic-create:
	kafka-topics --bootstrap-server=localhost:9092 --create --topic=my-topic --partitions=3 --replication-factor=1
kafka-topic-delete:
	kafka-topics --bootstrap-server=localhost:9092 --delete --topic=my-topic
kafka-topic-list:
	kafka-topics --bootstrap-server=localhost:9092 --list
kafka-topic-describe:
	kafka-topics --bootstrap-server=localhost:9092 --topic=my-topic --describe

kafka-console-producer:
	kafka-console-producer --bootstrap-server=localhost:9092 --topic=my-topic
kafka-console-consumer:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic
kafka-console-consumer-from-beginning:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --from-beginning
kafka-console-consumer-group:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --group=my-group

kafka-consumer-group-list:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --list
kafka-consumer-group-describe:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --describe --group=my-group
kafka-consumer-group-reset-offset-to-earliest:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --group=my-group --topic=my-topic --reset-offsets --to-earliest --execute
kafka-consumer-group-reset-offset-shift-by:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --group=my-group --topic=my-topic --reset-offsets --shift-by=-1 --execute
