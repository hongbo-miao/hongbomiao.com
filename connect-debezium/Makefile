# Docker
docker-build:
	cd .. && \
	docker build --file=connect-debezium/Dockerfile --tag=hongbomiao/hm-connect-debezium:latest .
docker-push:
	docker push hongbomiao/hm-connect-debezium:latest

# Debezium
debezium-get-roles-connector:
	kubectl --namespace=kafka get kctr roles-connector --output=yaml
debezium-list-topics:
	kubectl --namespace=kafka exec my-cluster-kafka-0 -c kafka -i -t -- \
	  bin/kafka-topics.sh \
	    --bootstrap-server localhost:9092 \
	    --list
debezium-consume:
	kubectl --namespace=kafka exec my-cluster-kafka-0 -c kafka -i -t -- \
	  bin/kafka-console-consumer.sh \
		--bootstrap-server localhost:9092 \
		--topic opa_db_server.public.roles
debezium-consume-from-beginning:
	kubectl --namespace=kafka exec my-cluster-kafka-0 -c kafka -i -t -- \
	  bin/kafka-console-consumer.sh \
		--bootstrap-server localhost:9092 \
		--topic opa_db_server.public.roles \
		--from-beginning
