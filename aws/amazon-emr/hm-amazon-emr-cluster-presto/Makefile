presto-status:
	sudo systemctl status presto-server
presto-start:
	sudo systemctl start presto-server
presto-stop:
	sudo systemctl stop presto-server

presto-show-presto-config:
	sudo cat /etc/presto/conf/config.properties

presto-execute:
	presto-cli \
		--server=localhost:8889 \
		--execute='show catalogs;'
	presto-cli \
		--server=localhost:8889 \
		--catalog=system \
		--execute='show schemas;'
	presto-cli \
		--server=localhost:8889 \
		--catalog=awsdatacatalog \
		--execute='show schemas;'
	presto-cli \
		--server=localhost:8889 \
		--catalog=awsdatacatalog \
		--schema=information_schema \
		--execute='show tables;'
