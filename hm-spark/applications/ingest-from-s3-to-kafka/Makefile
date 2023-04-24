sbt-reload:
	sbt reload
sbt-clean:
	sbt clean
sbt-compile:
	sbt compile
sbt-package:
	sbt package
sbt-test:
	sbt test
sbt-run:
	sbt run \
		-Dspark.hadoop.fs.s3a.access.key=xxx \
		-Dspark.hadoop.fs.s3a.secret.key=xxx
sbt-assembly:
	sbt assembly

sbt-clean-compile-package:
	sbt clean compile package
sbt-clean-compile-assembly:
	sbt clean compile assembly

sbt-plugins:
	sbt plugins

lint-scala:
	sbt scalafmtCheckAll
lint-scala-fix:
	sbt scalafmtAll

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

# 2 - Run in a pod
docker-build:
	cd ../../.. && \
	docker build --file=hm-spark/applications/ingest-from-s3-to-kafka/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-ingest-from-s3-to-kafka:latest .
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
        --conf=spark.executor.instances=5 \
        --conf=spark.kubernetes.namespace=hm-spark \
        --conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-ingest-from-s3-to-kafka:latest \
        --conf=spark.kubernetes.container.image.pullPolicy=Always \
        --conf=spark.hadoop.fs.s3a.access.key=xxx \
        --conf=spark.hadoop.fs.s3a.secret.key=xxx \
        local:///opt/spark/work-dir/IngestFromS3ToKafka-assembly-1.0.jar
kubectl-delete-spark-applications-in-kubernetes-cluster:
	kubectl delete pods --namespace=hm-spark --selector=spark-app-name=ingest-from-s3-to-kafka

# 3 - Spark standalone mode
serve-files:
	brew install codeskyblue/tap/gohttpserver
	gohttpserver \
		--root=hm-spark/applications/ \
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
			https://xxx.ngrok-free.app/ingest-from-s3-to-kafka/target/scala-2.12/IngestFromS3ToKafka-assembly-1.0.jar

# 3.2 - Client mode
spark-submit-to-spark-master-node-client-mode:
	# Note: update xxx
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
 		spark-submit \
			--master=spark://spark-master-svc.hm-spark.svc:7077 \
			--deploy-mode=client \
			--class=com.hongbomiao.IngestFromS3ToKafka \
			https://xxx.ngrok-free.app/ingest-from-s3-to-kafka/target/scala-2.12/IngestFromS3ToKafka-assembly-1.0.jar
