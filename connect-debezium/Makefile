# Docker
docker-build:
	cd .. && \
	docker build --file=connect-debezium/Dockerfile --tag=hongbomiao/hm-connect-debezium:latest .
docker-push:
	docker push hongbomiao/hm-connect-debezium:latest

# Kafka
kafka-get-hm-connect-cluster:
	kubectl get kafkaconnects hm-connect-cluster --namespace=kafka --output=yaml
kafka-get-roles-connector:
	kubectl get kafkaconnectors roles-connector --namespace=kafka --output=yaml
kafka-list-topics:
	kubectl exec my-cluster-kafka-0 --namespace=kafka --container=kafka --stdin --tty -- \
	  bin/kafka-topics.sh \
	    --bootstrap-server=localhost:9092 \
	    --list
kafka-consume:
	kubectl exec my-cluster-kafka-0 --namespace=kafka --container=kafka --stdin --tty -- \
	  bin/kafka-console-consumer.sh \
		--bootstrap-server=localhost:9092 \
		--topic=opa_db_server.public.roles
kafka-consume-from-beginning:
	kubectl exec my-cluster-kafka-0 --namespace=kafka --container=kafka --stdin --tty -- \
	  bin/kafka-console-consumer.sh \
		--bootstrap-server=localhost:9092 \
		--topic=opa_db_server.public.roles \
		--from-beginning
