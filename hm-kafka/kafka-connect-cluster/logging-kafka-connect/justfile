# Docker
docker-build:
	cd ../.. && \
	docker build --file=hm-kafka/kafka-connect-cluster/logging-kafka-connect/Dockerfile --tag=ghcr.io/hongbo-miao/hm-kafka-logging-kafka-connect:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-kafka-logging-kafka-connect:latest
docker-sh:
	docker run -it ghcr.io/hongbo-miao/hm-kafka-logging-kafka-connect:latest sh

# Kubernetes
kafka-get-hm-kafka-logging-kafka-connect:
	kubectl get kafkaconnects hm-kafka-logging-kafka-connect --namespace=hm-kafka --output=yaml

kafka-get-postgres-kafka-connector:
	kubectl get kafkaconnectors postgres-kafka-connector --namespace=hm-kafka --output=yaml
kafka-get-hm-dummy-log-elasticsearch-sink-kafka-connector:
	kubectl get kafkaconnectors hm-dummy-log-elasticsearch-sink-kafka-connector --namespace=hm-kafka --output=yaml
kafka-get-http-sink-kafka-connector:
	kubectl get kafkaconnectors http-sink-kafka-connector --namespace=hm-kafka --output=yaml

kafka-consume:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=hm.dummy-log \
			--property=print.key=true
kafka-consume-from-beginning:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=hm.dummy-log \
			--property=print.key=true \
			--from-beginning
