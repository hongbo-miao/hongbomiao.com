sbt-reload:
	sbt reload
sbt-clean:
	sbt clean
sbt-compile:
	sbt compile
sbt-package:
	sbt package
sbt-assembly:
	sbt assembly
sbt-clean-compile-package:
	sbt clean compile package
sbt-clean-compile-assembly:
	sbt clean compile assembly

sbt-test:
	sbt test
sbt-run:
	sbt run \
		-Dspark.hadoop.fs.s3a.access.key=xxx \
		-Dspark.hadoop.fs.s3a.secret.key=xxx

sbt-plugins:
	sbt plugins

lint-scala-scalafmt:
	sbt scalafmtCheckAll
lint-scala-scalafmt-fix:
	sbt scalafmtAll
lint-scala-scalafix:
	sbt "scalafixAll --check"
lint-scala-scalafix-fix:
	sbt scalafixAll

java-list-versions:
	/usr/libexec/java_home -V
java-current-version:
	java -version
java-set-version:
	export JAVA_HOME=`/usr/libexec/java_home -v 11.0.17`

# 1 - Local mode
spark-submit-to-local:
	spark-submit \
		--class=com.hongbomiao.IngestFromS3ToKafka \
		--master="local[*]" \
		target/scala-2.12/IngestFromS3ToKafka-assembly-1.0.jar

# 2 - Run in a pod. Cluster mode (the master node will assign to worker nodes)
rclone-sync:
	cd ../../.. && \
	rclone sync --progress hardware-in-the-loop/national-instruments/hm-tdms/data hm-s3:hm-production-bucket/iot --include="*.parquet"

kafka-consume:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-console-consumer.sh \
			--bootstrap-server=localhost:9092 \
			--topic=hm.motor \
			--property=print.key=true
kafka-topic-delete:
	kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
		bin/kafka-topics.sh \
			--bootstrap-server=localhost:9092 \
			--delete \
			--topic=hm.motor

kubectl-apply-telegraf:
	cd ../../.. && \
	kubectl apply --filename=kubernetes/manifests/telegraf
influx-measurement-count:
	export INFLUX_TOKEN=xxx && \
	echo 'from(bucket: "primary") |> range(start: 2010-01-01T00:00:00Z, stop: 2030-01-01T00:00:00Z) |> filter(fn: (r) => r["_measurement"] == "motor") |> count() |> yield(name: "count")' | \
		influx query --org=primary -
influx-measurement-truncate:
	export INFLUX_TOKEN=xxx && \
	influx delete \
		--org=primary \
		--bucket=primary \
		--start=2010-01-01T00:00:00Z \
		--stop=2030-01-01T00:00:00Z

kubectl-apply-hm-kafka-iot-kafka-connect:
	cd ../../.. && \
	kubectl apply --filename=kubernetes/manifests/hm-kafka/iot-kafka-connect
timescaledb-table-count:
	psql postgresql://admin:passw0rd@localhost:25495/production_hm_iot_db --command="select count(*) from motor;"
timescaledb-table-truncate:
	psql postgresql://admin:passw0rd@localhost:25495/production_hm_iot_db --command="truncate table motor;"

docker-build:
	cd ../../../.. && \
	docker build --file=data-processing/hm-spark/applications/ingest-from-s3-to-kafka/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-ingest-from-s3-to-kafka:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-spark-ingest-from-s3-to-kafka:latest
kubectl-cluster-info:
	kubectl cluster-info
spark-submit-to-kubernetes-cluster:
	spark-submit \
		--master=k8s://https://127.0.0.1:6443 \
		--deploy-mode=cluster \
		--name=ingest-from-s3-to-kafka \
		--class=com.hongbomiao.IngestFromS3ToKafka \
		--conf=spark.kubernetes.driverEnv.SPARK_MASTER_URL=spark://spark-master-svc.hm-spark.svc:7077 \
		--conf=spark.kubernetes.namespace=hm-spark \
		--conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-ingest-from-s3-to-kafka:latest \
		--conf=spark.kubernetes.container.image.pullPolicy=Always \
		--conf=spark.hadoop.fs.s3a.access.key=xxx \
		--conf=spark.hadoop.fs.s3a.secret.key=xxx \
		local:///opt/spark/work-dir/IngestFromS3ToKafka-assembly-1.0.jar
spark-submit-to-kubernetes-cluster-delight:
	spark-submit \
		--master=k8s://https://127.0.0.1:6443 \
		--deploy-mode=cluster \
		--name=ingest-from-s3-to-kafka \
		--class=com.hongbomiao.IngestFromS3ToKafka \
		--repositories=https://oss.sonatype.org/content/repositories/snapshots \
		--packages=co.datamechanics:delight_2.12:latest-SNAPSHOT \
		--conf=spark.kubernetes.namespace=hm-spark \
		--conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-ingest-from-s3-to-kafka:latest \
		--conf=spark.kubernetes.container.image.pullPolicy=Always \
		--conf=spark.hadoop.fs.s3a.access.key=xxx \
		--conf=spark.hadoop.fs.s3a.secret.key=xxx \
		--conf=spark.extraListeners=co.datamechanics.delight.DelightListener \
		--conf=spark.delight.accessToken.secret=xxx \
		local:///opt/spark/work-dir/IngestFromS3ToKafka-assembly-1.0.jar
kubectl-delete-spark-applications-in-kubernetes-cluster:
	kubectl delete pods --namespace=hm-spark --selector=spark-app-name=ingest-from-s3-to-kafka

# 3 - Spark standalone mode
serve-files:
	brew install codeskyblue/tap/gohttpserver
	gohttpserver \
		--root=data-processing/hm-spark/applications/ \
		--port=32609 \
		--upload
	ngrok http 32609

# 3.1 - Cluster mode (the master node will assign to worker nodes)
spark-submit-to-spark-master-node-cluster-mode:
	# Note: update xxx
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
		spark-submit \
			--master=spark://spark-master-svc.hm-spark.svc:7077 \
			--deploy-mode=cluster \
			--class=com.hongbomiao.IngestFromS3ToKafka \
			--conf=spark.hadoop.fs.s3a.access.key=xxx \
			--conf=spark.hadoop.fs.s3a.secret.key=xxx \
			https://xxx.ngrok-free.app/ingest-from-s3-to-kafka/target/scala-2.12/IngestFromS3ToKafka-assembly-1.0.jar
			--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,org.apache.spark:spark-avro_2.12:3.3.2 \

# 3.2 - Client mode
spark-submit-to-spark-master-node-client-mode:
	# Note: update xxx
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
		spark-submit \
			--master=spark://spark-master-svc.hm-spark.svc:7077 \
			--deploy-mode=client \
			--class=com.hongbomiao.IngestFromS3ToKafka \
			--conf=spark.hadoop.fs.s3a.access.key=xxx \
			--conf=spark.hadoop.fs.s3a.secret.key=xxx \
			https://xxx.ngrok-free.app/ingest-from-s3-to-kafka/target/scala-2.12/IngestFromS3ToKafka-assembly-1.0.jar
			--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,org.apache.spark:spark-avro_2.12:3.3.2 \
