sbt-reload:
	sbt reload
sbt-compile:
	sbt compile
sbt-package:
	sbt package
sbt-test:
	sbt test
sbt-run:
	sbt run

docker-build:
	cd ../../.. && \
	docker build --file=hm-spark/applications/find-retired-people-scala/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest

kubectl-proxy:
	kubectl proxy

spark-submit-local:
	spark-submit \
		--class=com.sundogsoftware.spark.FindRetiredPeople \
		--master="local[*]" \
		target/scala-2.12/findretiredpeople_2.12-1.0.jar
spark-submit-kubernetes:
	spark-submit \
        --master=k8s://https://127.0.0.1:6443 \
        --deploy-mode=cluster \
        --name=find-retired-people \
		--class=com.sundogsoftware.spark.FindRetiredPeople \
        --conf=spark.executor.instances=5 \
        --conf=spark.kubernetes.namespace=hm-spark \
        --conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest \
        local:///opt/spark/work-dir/findretiredpeople_2.13-1.0.jar

java-list-versions:
	/usr/libexec/java_home -V
java-current-version:
	java version
java-set-version:
	export JAVA_HOME=`/usr/libexec/java_home -v 17.0.6`
