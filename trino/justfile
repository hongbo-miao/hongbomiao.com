trino-docker-run:
	docker run --publish=44116:8080 --name=hm-trino trinodb/trino
trino-docker-rm:
	docker rm hm-trino

trino-docker-exec-trino:
	docker exec --interactive --tty hm-trino \
		trino
trino-docker-exec-trino-catalog-schema:
	docker exec --interactive --tty hm-trino \
		trino \
			--catalog=tpch \
			--schema=sf1

trino-docker-exec-trino-select:
	docker exec --interactive --tty hm-trino \
		trino \
			--execute='select nationkey, name, regionkey from tpch.sf1.nation limit 5'
trino-docker-exec-trino-catalog-schema-select:
	docker exec --interactive --tty hm-trino \
		trino \
			--catalog=tpch \
			--schema=sf1 \
			--execute='select nationkey, name, regionkey from nation limit 5'

trino-docker-exec-trino-debug:
	docker exec --interactive --tty hm-trino \
		trino --debug
