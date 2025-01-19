docker-fluent-bit-help:
	docker run --interactive --tty --rm cr.fluentbit.io/fluent/fluent-bit:2.0.11 \
		--help

docker-fluent-bit-dummy:
	docker run \
		--rm \
		cr.fluentbit.io/fluent/fluent-bit:2.0.11 \
			--input=dummy \
			--output=stdout \
			--flush=1
