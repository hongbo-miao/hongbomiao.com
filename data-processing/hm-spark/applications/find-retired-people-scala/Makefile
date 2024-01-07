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
	sbt run

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
	export JAVA_HOME=`/usr/libexec/java_home -v 17.0.6`

# 1 - Local mode
spark-submit-to-local:
	spark-submit \
		--class=com.hongbomiao.FindRetiredPeople \
		--master="local[*]" \
		target/scala-2.12/FindRetiredPeople-assembly-1.0.jar

# 2 - Run in a pod. Cluster mode (the master node will assign to worker nodes)
docker-build:
	cd ../../../.. && \
	docker build --file=data-processing/hm-spark/applications/find-retired-people-scala/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest
kubectl-cluster-info:
	kubectl cluster-info
spark-submit-to-kubernetes-cluster:
	spark-submit \
        --master=k8s://https://127.0.0.1:6443 \
        --deploy-mode=cluster \
        --name=find-retired-people-scala \
		--class=com.hongbomiao.FindRetiredPeople \
		--conf=spark.kubernetes.driverEnv.SPARK_MASTER_URL=spark://spark-master-svc.hm-spark.svc:7077 \
        --conf=spark.kubernetes.namespace=hm-spark \
        --conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest \
        --conf=spark.kubernetes.container.image.pullPolicy=Always \
        local:///opt/spark/work-dir/FindRetiredPeople-assembly-1.0.jar
kubectl-delete-spark-applications-in-kubernetes-cluster:
	kubectl delete pods --namespace=hm-spark --selector=spark-app-name=find-retired-people-scala

# 3 - Spark standalone mode
serve-files:
	brew install codeskyblue/tap/gohttpserver
	gohttpserver \
		--root=data-processing/hm-spark/applications/find-retired-people-scala/target/scala-2.12/ \
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
			--class=com.hongbomiao.FindRetiredPeople \
			https://xxx.ngrok-free.app/find-retired-people-scala/target/scala-2.12/FindRetiredPeople-assembly-1.0.jar

# 3.2 - Client mode
spark-submit-to-spark-master-node-client-mode:
	# Note: update xxx
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
		spark-submit \
			--master=spark://spark-master-svc.hm-spark.svc:7077 \
			--deploy-mode=client \
			--class=com.hongbomiao.FindRetiredPeople \
			https://xxx.ngrok-free.app/find-retired-people-scala/target/scala-2.12/FindRetiredPeople-assembly-1.0.jar
