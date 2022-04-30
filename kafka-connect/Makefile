# Docker
docker-build:
	cd .. && \
	docker build --file=kafka-connect/Dockerfile --tag=ghcr.io/hongbo-miao/hm-kafka-connect:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-kafka-connect:latest
docker-sh:
	docker run -it hongbomiao/hm-kafka-connect:latest sh

# Kafka
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
