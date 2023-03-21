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
	sbt run

docker-build:
	cd ../../.. && \
	docker build --file=hm-spark/applications/find-retired-people-scala/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest

kubectl-cluster-info:
	kubectl cluster-info

local-spark-submit:
	spark-submit \
		--class=com.hongbomiao.FindRetiredPeople \
		--master="local[*]" \
		target/scala-2.12/findretiredpeople_2.12-1.0.jar

kubernetes-spark-submit:
	spark-submit \
        --master=k8s://https://127.0.0.1:6443 \
        --deploy-mode=cluster \
        --name=find-retired-people \
		--class=com.hongbomiao.FindRetiredPeople \
        --conf=spark.executor.instances=5 \
        --conf=spark.kubernetes.namespace=hm-spark \
        --conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-find-retired-people-scala:latest \
        --conf=spark.kubernetes.container.image.pullPolicy=Always \
        local:///opt/spark/work-dir/findretiredpeople_2.12-1.0.jar
kubernetes-delete-spark-applications:
	kubectl delete pods --namespace=hm-spark --selector=spark-app-name=find-retired-people

java-list-versions:
	/usr/libexec/java_home -V
java-current-version:
	java version
java-set-version:
	export JAVA_HOME=`/usr/libexec/java_home -v 17.0.6`
