generate-dummy-log:
	sh bin/generate-dummy-log.sh

fluent-bit:
	cd ../.. && \
	docker run \
		--rm \
        --volume="$$(pwd)/fluent-bit/write-dummy-log-to-kafka/fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf" \
        --volume="$$(pwd)/fluent-bit/write-dummy-log-to-kafka/data/dummy.log:/usr/src/app/dummy.log" \
        --volume="$$(pwd)/kubernetes/data/hm-kafka/hm-kafka/ca.crt:/usr/src/app/ca.crt" \
		cr.fluentbit.io/fluent/fluent-bit:2.0.11

kafka-console-consumer-from-beginning:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=hm.dummy-log \
			--from-beginning
