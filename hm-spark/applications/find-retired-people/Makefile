openjdk-install:
	brew install openjdk@17

poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

poetry-run-dev:
	poetry run poe dev

# 2 - Submit the application to the Kubernetes cluster
poetry-run-spark-submit-to-local:
	poetry run poe spark-submit-to-local

# 2 - Submit the application to the Kubernetes cluster
docker-build:
	cd ../../.. && \
	docker build --file=hm-spark/applications/find-retired-people/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-find-retired-people:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-spark-find-retired-people:latest
kubectl-cluster-info:
	kubectl cluster-info
spark-submit-to-kubernetes-cluster:
	spark-submit \
        --master=k8s://https://127.0.0.1:6443 \
        --deploy-mode=cluster \
        --name=find-retired-people \
		--class=com.hongbomiao.FindRetiredPeople \
        --conf=spark.executor.instances=5 \
        --conf=spark.kubernetes.namespace=hm-spark \
        --conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-find-retired-people:latest \
        --conf=spark.kubernetes.container.image.pullPolicy=Always \
        local:///opt/spark/work-dir/src/main.py
kubectl-delete-spark-applications-in-kubernetes-cluster:
	kubectl delete pods --namespace=hm-spark --selector=spark-app-name=find-retired-people

# 3 - Submit the application to Spark master node in the Kubernetes cluster
serve-jar-file:
	brew install codeskyblue/tap/gohttpserver
	gohttpserver \
		--root=src/ \
		--port=32609 \
		--upload
	ngrok http 32609

# 3.1 - Spark standalone cluster in cluster deploy mode (the master node will assign to worker nodes)
# (The standalone mode does not support cluster mode for Python applications)
# spark-submit-to-spark-master-node-cluster-mode:
# 	# Note: update xxx
# 	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
#  		spark-submit \
# 			--master=spark://spark-master-svc.hm-spark.svc:7077 \
# 			--deploy-mode=cluster \
# 			https://xxx.ngrok.io/main.py

# 3.2 - Spark standalone cluster in client deploy mode (run in the master node)
spark-submit-to-spark-master-node-client-mode:
	# Note: update xxx
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
 		spark-submit \
			--master=spark://spark-master-svc.hm-spark.svc:7077 \
			--deploy-mode=client \
			https://xxx.ngrok.io/main.py
