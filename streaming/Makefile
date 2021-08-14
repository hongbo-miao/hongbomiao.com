# Maven
maven-install:
	mvn install
maven-compile:
	mvn compile
maven-test:
	mvn test
maven-verify:
	mvn verify
maven-package:
	mvn clean package

# Flink
flink-local-start:
	/usr/local/Cellar/apache-flink/1.13.1/libexec/bin/start-cluster.sh
flink-local-stop:
	/usr/local/Cellar/apache-flink/1.13.1/libexec/bin/stop-cluster.sh

flink-session-start:
	/usr/local/Cellar/apache-flink/1.13.1/libexec/bin/kubernetes-session.sh \
        -Dkubernetes.cluster-id=hm-streaming-cluster
flink-session-ui:
	minikube tunnel
flink-session-submit-job:
	flink run \
        --target kubernetes-session \
        -Dkubernetes.cluster-id=hm-streaming-cluster \
        ./target/streaming-0.1.jar

flink-application-start:
	flink run-application \
		--target kubernetes-application \
		-Dkubernetes.cluster-id=hm-streaming-cluster \
		-Dkubernetes.container.image=hongbomiao/hm-streaming:latest \
		local:///opt/flink/usrlib/streaming-0.1.jar

flink-delete:
	kubectl delete deployment/hm-streaming-cluster
