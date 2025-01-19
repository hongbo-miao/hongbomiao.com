uv-install-python:
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev

uv-run-dev:
	uv run poe dev
uv-run-test:
	uv run poe test
uv-run-test-coverage:
	uv run poe test-coverage

# 1 - Local mode
spark-submit-to-local:
	uv run poe spark-submit-to-local

# 2 - Run in a pod. Cluster mode (the master node will assign to worker nodes)
docker-build:
	cd ../../../.. && \
	docker build --file=data-processing/hm-spark/applications/find-retired-people-python/Dockerfile --tag=ghcr.io/hongbo-miao/hm-spark-find-retired-people-python:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-spark-find-retired-people-python:latest
kubectl-cluster-info:
	kubectl cluster-info
spark-submit-to-kubernetes-cluster:
	spark-submit \
		--master=k8s://https://127.0.0.1:6443 \
		--deploy-mode=cluster \
		--name=find-retired-people-python \
		--conf=spark.kubernetes.driverEnv.SPARK_MASTER_URL=spark://spark-master-svc.hm-spark.svc:7077 \
		--conf=spark.kubernetes.namespace=hm-spark \
		--conf=spark.kubernetes.container.image=ghcr.io/hongbo-miao/hm-spark-find-retired-people-python:latest \
		--conf=spark.kubernetes.container.image.pullPolicy=Always \
		local:///opt/spark/work-dir/src/main.py
kubectl-delete-spark-applications-in-kubernetes-cluster:
	kubectl delete pods --namespace=hm-spark --selector=spark-app-name=find-retired-people-python

# 3 - Spark standalone mode
serve-files:
	brew install codeskyblue/tap/gohttpserver
	gohttpserver \
		--root=src/ \
		--port=32609 \
		--upload
	ngrok http 32609

# 3.1 - Cluster mode (the master node will assign to worker nodes)
# (The standalone mode does not support cluster mode for Python applications)
# spark-submit-to-spark-master-node-cluster-mode:
# 	# Note: update xxx
# 	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
# 		spark-submit \
# 			--master=spark://spark-master-svc.hm-spark.svc:7077 \
# 			--deploy-mode=cluster \
# 			https://xxx.ngrok-free.app/find-retired-people-python/src/main.py?download=true

# 3.2 - Client mode
spark-submit-to-spark-master-node-client-mode:
	# Note: update xxx
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
		spark-submit \
			--master=spark://spark-master-svc.hm-spark.svc:7077 \
			--deploy-mode=client \
			https://xxx.ngrok-free.app/find-retired-people-python/src/main.py?download=true
