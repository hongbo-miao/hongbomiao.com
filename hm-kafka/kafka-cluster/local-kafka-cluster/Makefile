kafka-install:
	brew install kafka
zookeeper-start:
	zookeeper-server-start /opt/homebrew/etc/kafka/zookeeper.properties
kafka-start:
	kafka-server-start /opt/homebrew/etc/kafka/server.properties
zookeeper-status:
	zkServer status

kafka-version:
	kafka-topics --version

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
kafka-topic-consume:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --property=print.key=true
kafka-topic-consume-from-beginning:
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
