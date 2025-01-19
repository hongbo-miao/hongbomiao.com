set-up:
	bash bin/set_up.sh
clean:
	bash bin/clean.sh

set-up-multipass:
	sh bin/set_up_multipass.sh
set-up-k3d:
	sh bin/set_up_k3d.sh
set-up-kind:
	sh bin/set_up_kind.sh
set-up-minikube:
	sh bin/set_up_minikube.sh

focus-opa-debug:
	sh bin/utils/focus_opa_debug.sh
focus-pixie-debug:
	sh bin/utils/focus_pixie_debug.sh
focus-rasa-debug:
	sh bin/utils/focus_rasa_debug.sh

install-alpine:
	sh bin/utils/install_hm_alpine.sh

# minikube
minikube-start-docker:
	minikube start --driver=docker --cpus=4 --memory=8g
minikube-start-hyperkit:
	minikube start --driver=hyperkit --cpus=4 --memory=8g
minikube-start-virtualbox:
	minikube start --driver=virtualbox --cpus=4 --memory=8g
minikube-enable-ingress:
	minikube addons enable ingress
minikube-mount:
	minikube mount ./data:/data
minikube-delete:
	minikube delete
minikube-dashboard:
	minikube dashboard
minikube-ip:
	minikube ip

# kind
kind-create:
	kind create cluster --name=west --config=kind/west-cluster-config.yaml
	kind create cluster --name=east --config=kind/east-cluster-config.yaml
kind-delete:
	kind delete cluster --name=west
	kind delete cluster --name=east
kind-list-clusters:
	kind get clusters

# k3d
k3d-cluster-create:
	k3d cluster create west --config=k3d/west-cluster-config.yaml
	k3d cluster create east --config=k3d/west-cluster-config.yaml
	k3d cluster create dev --config=k3d/dev-cluster-config.yaml
k3d-cluster-delete:
	k3d cluster delete west
	k3d cluster delete east
	k3d cluster delete dev
k3d-cluster-list:
	k3d cluster list
k3d-node-list:
	k3d node list

# Multipass
multipass-launch-node:
	multipass launch --name=west-master --cpus=6 --mem=24g --disk=256g
multipass-exec:
	multipass exec west-master -- bash
multipass-list:
	multipass list
multipass-mount:
	multipass mount "${PWD}/data" west-master:/data
multipass-info:
	multipass info west-master
multipass-start:
	multipass start west-master
multipass-stop:
	multipass stop west-master
multipass-delete:
	multipass delete west-master
multipass-purge:
	multipass purge

# nslookup
k8s-debug:
	nslookup graphql-server-service.hm.svc.cluster.local

# Kubernetes
kubectl-apply:
	kubectl apply --filename=manifests/west/hm-namespace.yaml
	kubectl apply --filename=manifests/west
	kubectl apply --filename=manifests/west/hm-limitrange.yaml
	kubectl apply --filename=manifests/west --selector=app.kubernetes.io/name=graphql-server
kubectl-apply-with-linkerd:
	linkerd inject - | kubectl apply --filename=manifests
kubectl-delete:
	kubectl delete --filename=manifests
kubectl-get-customer-resources:
	kubectl api-resources --api-group=gateway.networking.k8s.io
kubectl-delete-customer-resources:
	kubectl delete customresourcedefinition gatewayclasses.gateway.networking.k8s.io
	kubectl delete customresourcedefinition gateways.gateway.networking.k8s.io
	kubectl delete customresourcedefinition grpcroutes.gateway.networking.k8s.io
	kubectl delete customresourcedefinition httproutes.gateway.networking.k8s.io
	kubectl delete customresourcedefinition referencegrants.gateway.networking.k8s.io
kubectl-get-node:
	kubectl get node --output=wide
kubectl-get-pods-all:
	kubectl get pods --all-namespaces
kubectl-get-pods:
	kubectl get pods --namespace=hm
kubectl-get-services-all:
	kubectl get services --all-namespaces
kubectl-get-services:
	kubectl get services --namespace=hm
kubectl-get-services-yaml:
	kubectl get services graphql-server-service --namespace=hm --output=yaml
kubectl-get-deployments:
	kubectl get deployments --namespace=hm
kubectl-get-deployments-yaml:
	kubectl get deployments graphql-server-deployment --namespace=hm --output=yaml
kubectl-get-namespaces:
	kubectl get namespaces
kubectl-get-storageclasses:
	kubectl get storageclasses --all-namespaces
kubectl-get-persistentvolumes:
	kubectl get persistentvolumes --all-namespaces
kubectl-get-persistentvolumeclaims:
	kubectl get persistentvolumeclaims --all-namespaces
kubectl-get-endpoints:
	kubectl get endpoints graphql-server-service --namespace=hm
kubectl-get-configmaps-all:
	kubectl get configmaps --all-namespaces
kubectl-get-configmaps:
	kubectl get configmaps --namespace=hm
kubectl-get-configmaps-yaml:
	kubectl get configmaps ingress-nginx-controller --namespace=ingress-nginx --output=yaml
kubectl-get-serviceaccounts:
	kubectl get serviceaccounts
kubectl-get-apiservices:
	kubectl get apiservices
kubectl-logs:
	kubectl logs --follow POD_NAME --namespace=hm
kubectl-sh:
	kubectl exec --stdin --tty POD_NAME --namespace=hm -- sh
kubectl-rollout-restart:
	kubectl rollout restart hasura-deployment --namespace=hm-hasura
kubectl-list-prometheus-containers:
	kubectl get pod prometheus-monitoring-kube-prometheus-prometheus-0 --namespace=hm-monitoring --output=jsonpath='{.spec.containers[*].name}'
kubectl-list-alertmanager-containers:
	kubectl get pod alertmanager-monitoring-kube-prometheus-alertmanager-0 --namespace=hm-monitoring --output=jsonpath='{.spec.containers[*].name}'
kubectl-list-grafana-containers:
	kubectl get pod monitoring-grafana-xxx --namespace=hm-monitoring --output=jsonpath='{.spec.containers[*].name}'

kubectl-config-view:
	kubectl config view
kubectl-config-get-contexts:
	kubectl config get-contexts
kubectl-config-current-context:
	kubectl config current-context
kubectl-config-use-context:
	kubectl config use-context k3d-west
kubectl-cluster-info:
	kubectl cluster-info
kubectl-cluster-info-context:
	kubectl cluster-info --context k3d-west
kubectl-proxy:
	kubectl proxy

kubectl-port-forward-hasura:
	# hasura-graphql-engine
	kubectl port-forward service/hasura-service --namespace=hm-hasura 16020:16020
	# hasura-metric-adapter
	kubectl port-forward service/hasura-service --namespace=hm-hasura 16021:16021
	# traefik-hasura
	kubectl port-forward service/hasura-service --namespace=hm-hasura 16022:16022
	# traefik
	kubectl port-forward service/hasura-service --namespace=hm-hasura 16023:16023
kubectl-port-forward-graphql-server:
	# graphql-server
	kubectl port-forward service/graphql-server-service --namespace=hm 31800:31800
	# opa
	kubectl port-forward service/graphql-server-service --namespace=hm 8181:8181
	# opal-client
	kubectl port-forward service/graphql-server-service --namespace=hm 7000:7000
kubectl-port-forward-opal-server:
	kubectl port-forward service/opal-server-service --namespace=hm 7002:7002
kubectl-port-forward-config-loader:
	kubectl port-forward service/config-loader-service --namespace=hm 26660:26660
kubectl-port-forward-api-python:
	kubectl port-forward service/api-python-service --namespace=hm-api-python 35903:35903

kubectl-port-forward-clickhouse:
	kubectl port-forward service/chi-simple-01-simple-0-0 --namespace=hm-clickhouse 8123:8123
	kubectl port-forward service/chi-simple-01-simple-0-0 --namespace=hm-clickhouse 9000:9000
kubectl-port-forward-postgres-operator-ui:
	kubectl port-forward service/postgres-operator-ui --namespace=hm-postgres-operator 8081:80
kubectl-port-forward-postgres-cluster:
	kubectl port-forward service/hm-postgres-cluster-repl --namespace=hm-postgres 31270:5432
kubectl-port-forward-postgres:
	kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432
kubectl-port-forward-hydra:
	kubectl port-forward service/hydra-service --namespace=hm-hydra 26958:5432
kubectl-port-forward-timescale:
	kubectl port-forward service/timescale --namespace=hm-timescale 25495:5432
kubectl-port-forward-influxdb:
	kubectl port-forward service/influxdb --namespace=hm-influxdb 20622:8086
kubectl-port-forward-redis:
	kubectl port-forward service/redis-headless --namespace=hm-redis 6379:6379
kubectl-port-forward-trino:
	kubectl port-forward service/trino --namespace=hm-trino 26869:8080
kubectl-port-forward-superset:
	kubectl port-forward service/superset --namespace=hm-superset 22357:8088
kubectl-port-forward-dgraph:
	kubectl port-forward service/dgraph-dgraph-alpha --namespace=hm-dgraph 8080:8080
	kubectl port-forward service/dgraph-dgraph-zero --namespace=hm-dgraph 6080:6080
kubectl-port-forward-qdrant:
	kubectl port-forward service/qdrant --namespace=hm-qdrant 6333:6333
kubectl-port-forward-elastic:
	kubectl port-forward service/hm-kibana-kb-http --namespace=hm-elastic 5601:5601
	kubectl port-forward service/hm-elasticsearch-es-http --namespace=hm-elastic 9200:9200
kubectl-port-forward-kafka-broker:
	kubectl port-forward service/hm-kafka-kafka-bootstrap --namespace=hm-kafka 9092:9092
kubectl-port-forward-kafka-zookeeper:
	kubectl port-forward service/hm-kafka-zookeeper-client --namespace=hm-kafka 2181:2181
kubectl-port-forward-hm-kafka-iot-kafka-connect:
	kubectl port-forward service/hm-kafka-iot-kafka-connect-connect-api --namespace=hm-kafka 8083:8083
kubectl-port-forward-hm-kafka-logging-kafka-connect:
	kubectl port-forward service/hm-kafka-logging-kafka-connect-connect-api --namespace=hm-kafka 8083:8083
kubectl-port-forward-hm-kafka-opa-kafka-connect:
	kubectl port-forward service/hm-kafka-opa-kafka-connect-connect-api --namespace=hm-kafka 8083:8083
kubectl-port-forward-apicurio-registry:
	kubectl port-forward service/apicurio-registry-apicurio-registry --namespace=hm-apicurio-registry 35185:8080
kubectl-port-forward-ui-for-apache-kafka:
	kubectl port-forward service/ui-for-apache-kafka-kafka-ui --namespace=hm-ui-for-apache-kafka 26917:80
kubectl-port-forward-redpanda-console:
	kubectl port-forward service/redpanda-console --namespace=hm-redpanda-console 46270:8080
kubectl-port-forward-akhq:
	kubectl port-forward service/akhq --namespace=hm-akhq 34594:80
kubectl-port-forward-spark:
	kubectl port-forward service/spark-master-svc --namespace=hm-spark 4040:80
	kubectl port-forward service/spark-master-svc --namespace=hm-spark 7077:7077
kubectl-port-forward-flink:
	kubectl port-forward service/hm-flink-cluster-rest --namespace=hm-flink 8081:8081
kubectl-port-forward-prefect-server:
	kubectl port-forward service/prefect-server --namespace=hm-prefect 4200:4200
kubectl-port-forward-prefect-server-postgresql:
	kubectl port-forward service/prefect-server-postgresql --namespace=hm-prefect 21945:5432
kubectl-port-forward-airflow:
	kubectl port-forward service/airflow-webserver --namespace=hm-airflow 45808:8080
kubectl-port-forward-prometheus:
	kubectl port-forward service/monitoring-kube-prometheus-prometheus --namespace=hm-monitoring 9090:9090
kubectl-port-forward-grafana-by-loki:
	kubectl port-forward service/grafana --namespace=hm-grafana 18845:80
kubectl-port-forward-netdata:
	kubectl port-forward service/netdata --namespace=hm-netdata 55839:19999
kubectl-port-forward-ory-hydra:
	kubectl port-forward service/ory-hydra-public --namespace=hm-ory-hydra 4444:4444
	kubectl port-forward service/ory-hydra-admin --namespace=hm-ory-hydra 4445:4445
kubectl-port-forward-kubeflow:
	kubectl port-forward service/ml-pipeline-ui --namespace=kubeflow 38620:80
kubectl-port-forward-mlflow:
	kubectl port-forward service/mlflow --namespace=hm-mlflow 29858:5000
kubectl-port-forward-rasa:
	kubectl port-forward service/hm-release-rasa --namespace=hm-rasa 39143:5005
	kubectl port-forward service/hm-release-postgresql-headless --namespace=hm-rasa 35785:5432
	kubectl port-forward service/hm-release-redis-master --namespace=hm-rasa 6379:6379
	kubectl port-forward service/hm-release-rasa-action-server --namespace=hm-rasa-action-server 5055:5055
kubectl-port-forward-kubecost:
	kubectl port-forward service/kubecost-cost-analyzer --namespace=hm-kubecost 9090:9090
kubectl-port-forward-polaris:
	kubectl port-forward service/polaris-dashboard --namespace=hm-polaris 31450:80
kubectl-port-forward-sloop:
	kubectl port-forward service/sloop --namespace=hm-sloop 16850:16850
kubectl-port-forward-rancher:
	kubectl port-forward service/rancher --namespace=cattle-system 46271:443
kubectl-port-forward-temporal:
	kubectl port-forward service/temporal-web --namespace=hm-temporal 48780:8080
kubectl-port-forward-metabase:
	kubectl port-forward service/metabase --namespace=hm-metabase 50399:80
kubectl-port-forward-pgadmin:
	kubectl port-forward service/pgadmin-pgadmin4 --namespace=hm-pgadmin 40594:80

kubectl-label-add:
	kubectl label namespace production-hm-goldilocks goldilocks.fairwinds.com/enabled=true
kubectl-label-remove:
	kubectl label namespace production-hm-goldilocks goldilocks.fairwinds.com/enabled-

list-port-forward:
	ps -ef | grep port-forward
kill-port:
	kill -9 PID

# K9s
k9s-install:
	brew install k9s
k9s:
	k9s

# kubectl-plugins
# https://github.com/luksa/kubectl-plugins
kubectl-ssh:
	kubectl ssh node lima-rancher-desktop

# Vagrant
vagrant-install:
	brew install --cask virtualbox
	brew install --cask vagrant
	brew install --cask vagrant-manager
vagrant-init:
	vagrant init ubuntu/jammy64
vagrant-up:
	vagrant up
vagrant-suspend:
	vagrant suspend
vagrant-halt:
	vagrant halt
vagrant-destroy:
	vagrant destroy --force
vagrant-status:
	vagrant status
vagrant-ssh:
	vagrant ssh
vagrant-help:
	vagrant help
vagrant-box-list:
	vagrant box list
vagrant-box-outdated:
	vagrant box outdated
vagrant-box-update:
	vagrant box update
vagrant-box-add:
	vagrant box add ubuntu/jammy64
vagrant-box-remove:
	vagrant box remove ubuntu/jammy64
vagrant-box-help:
	vagrant help box
vagrant-plugin-list:
	vagrant plugin list
vagrant-plugin-install:
	vagrant plugin install vagrant-vbguest
vagrant-plugin-uninstall:
	vagrant plugin uninstall vagrant-vbguest
vagrant-plugin-help:
	vagrant help plugin

# Skaffold
skaffold:
	skaffold dev

# Linkerd
linkerd-install-control-plane:
	linkerd install --values=linkerd/config.yaml --disable-heartbeat | kubectl apply --filename=-
linkerd-install-control-plane-prod:
	linkerd install --values=linkerd/config.yaml --disable-heartbeat --ha | kubectl apply --filename=-
linkerd-install-viz:
	linkerd viz install --set=jaegerUrl=jaeger.linkerd-jaeger:16686 | kubectl apply --filename=-
linkerd-install-jaeger:
	linkerd jaeger install | kubectl apply --filename=-
linkerd-viz-dashboard:
	linkerd viz dashboard &
linkerd-jaeger-dashboard:
	linkerd jaeger dashboard &
linkerd-get-yaml:
	linkerd install --disable-heartbeat > linkerd.yaml
linkerd-inject:
	kubectl get deployments --namespace=hm --output=yaml | linkerd inject - | kubectl apply --filename=-
linkerd-inject-nginx-controller:
	kubectl get deployment ingress-nginx-controller --namespace=ingress-nginx --output=yaml | linkerd inject --ingress - | kubectl apply --filename=-
linkerd-verify-inject-nginx-controller:
	kubectl describe pods/ingress-nginx-controller-xxx --namespace=ingress-nginx | grep "linkerd.io/inject: ingress"
linkerd-check:
	linkerd check
linkerd-check-pre:
	linkerd check --pre
linkerd-check-proxy: # includes linkerd-identity-data-plane
	linkerd check --proxy
linkerd-viz-tap:
	linkerd viz tap deployments/graphql-server-deployment --namespace=hm
linkerd-viz-tap-json:
	linkerd viz tap deployments/graphql-server-deployment --namespace=hm --output=json
linkerd-viz-tap-to:
	linkerd viz tap deployments/graphql-server-deployment --namespace=hm --to=deployment/grpc-server-deployment
linkerd-viz-tap-to-path:
	linkerd viz tap deployments/graphql-server-deployment --namespace=hm --to=deployment/grpc-server-deployment --path=/api.proto.greet.v1.GreetService/Greet
linkerd-viz-top: # shows traffic routes sorted by the most popular paths
	linkerd viz top deployments/graphql-server-deployment --namespace=hm
linkerd-viz-stat-deployments:
	linkerd viz stat deployments --namespace=hm
linkerd-viz-stat-trafficsplit:
	linkerd viz stat trafficsplit --context=k3d-west --namespace=hm
linkerd-viz-stat-wide: # includes extra READ_BYTES/SEC and WRITE_BYTES/SEC
	linkerd viz stat deployments --namespace=hm --output=wide
linkerd-viz-stat-from-to:
	linkerd viz stat --namespace=hm deployments/graphql-server-deployment --to deployments/grpc-server-deployment
linkerd-viz-stat-all-from: # views the metrics for traffic to all deployments that comes from graphql-server-deployment
	linkerd viz stat --namespace=hm deployments --from deployments/graphql-server-deployment
linkerd-viz-edges-deployments:
	linkerd viz edges deployments --namespace=hm
linkerd-viz-edges-deployments-json:
	linkerd viz edges deployments --namespace=hm --output=json
linkerd-viz-edges-pods:
	linkerd viz edges pods --namespace=hm
linkerd-viz-edges-pods-json:
	linkerd viz edges pods --namespace=hm --output=json
linkerd-viz-routes:
	linkerd viz routes deployments/graphql-server-deployment --namespace=hm
linkerd-viz-routes-wide: # includes EFFECTIVE_SUCCESS, EFFECTIVE_RPS, ACTUAL_SUCCESS, ACTUAL_RPS
	linkerd viz routes deployments/graphql-server-deployment --namespace=hm --to deployments/grpc-server-deployment --output=wide
linkerd-viz-routes-json:
	linkerd viz routes deployments/graphql-server-deployment --namespace=hm --output=json
linkerd-get-secrets:
	kubectl get secrets --namespace=linkerd
linkerd-get-secret-yaml:
	kubectl get secrets --namespace=linkerd linkerd-identity-issuer --output=yaml

# Pixie
pixie-login:
	px auth login
pixie-deploy:
	px deploy --y
pixie-delete:
	px delete

# Kibana
kibana-ui:
	kubectl port-forward service/hm-kb-http --namespace=hm-elastic 5601:5601 &
kibana-get-password: # username: elastic
	kubectl get secret hm-elasticsearch-es-elastic-user --namespace=hm-elastic --output=jsonpath="{.data.elastic}" | base64 --decode; echo

# Argo CD
argo-cd-install:
	kubectl create namespace argocd
	kubectl apply --namespace=argocd --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
argo-cd-ui:
	kubectl port-forward service/argocd-server --namespace=argocd 31026:443 &
argo-cd-get-password: # username: admin
	kubectl get secret argocd-initial-admin-secret --namespace=argocd --output=jsonpath="{.data.password}" | base64 -d && echo
argo-cd-login:
	$(eval ARGOCD_PASSWORD := $(shell kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo))
	argocd login localhost:31026 --username=admin --password=$(ARGOCD_PASSWORD) --insecure
argo-cd-enable-auth-sync:
	argocd app set hm-application --sync-policy=automated
argo-cd-disable-auth-sync:
	argocd app set hm-application --sync-policy=none
argo-cd-diff:
	argocd app diff hm-application --local=manifests
argo-cd-apply:
	kubectl apply --filename=manifests/argocd/hm-application.yaml
argo-cd-sync:
	kubectl apply --filename=manifests/argocd/hm-application.yaml
	argocd app sync hm-application --grpc-web
argo-cd-sync-local:
	argocd app sync hm-application --grpc-web --local=manifests/west
argo-cd-sync-local-full:
	kubectl create namespace hm
	ELASTIC_APM_TOKEN=$(kubectl get secret hm-apm-apm-token --namespace=hm-elastic --output=go-template='{{index .data "secret-token" | base64decode}}')
	kubectl create secret generic hm-elastic-apm --namespace=hm --from-literal="token=${ELASTIC_APM_TOKEN}"
	kubectl apply --filename=manifests/argocd/hm-application.yaml
	argocd app sync hm-application --grpc-web --local=manifests/west
argo-cd-list:
	argocd app list
argo-cd-delete:
	argocd app delete production-hm-application --yes
kubectl-get-pods-argocd:
	kubectl get pods --namespace=argocd
kubectl-get-rolebindings-argocd:
	kubectl get rolebindings --namespace=argocd
kubectl-describe-rolebinding-argocd:
	kubectl describe rolebinding argocd-application-controller --namespace=argocd
kubectl-get-roles-argocd:
	kubectl get roles --namespace=argocd
kubectl-describe-role-argocd:
	kubectl describe role argocd-application-controller --namespace=argocd

# Dgraph
dgraph-install-standalone:
	mkdir -p ~/dgraph
	docker run \
		--interactive \
		--tty \
		--name=dgraph \
		--publish=5080:5080 \
		--publish=6080:6080 \
		--publish=8080:8080 \
		--publish=9080:9080 \
		--publish=8000:8000 \
		--volume=~/dgraph:/dgraph \
		dgraph/standalone
dgraph-install:
	kubectl apply --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/dgraph-single/dgraph-single.yaml
dgraph-delete:
	kubectl delete --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/dgraph-single/dgraph-single.yaml
	kubectl delete persistentvolumeclaims --namespace=hm --selector=app=dgraph
dgraph-install-ha:
	kubectl apply --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/dgraph-ha/dgraph-ha.yaml
dgraph-delete-ha:
	kubectl delete --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/dgraph-ha/dgraph-ha.yaml
	kubectl delete persistentvolumeclaims --namespace=hm --selector=app=dgraph-zero
	kubectl delete persistentvolumeclaims --namespace=hm --selector=app=dgraph-alpha

# MinIO
# https://docs.min.io/docs/minio-client-complete-guide.html
mc-ls:
	mc ls
mc-alias-set:
	mc alias set hm https://localhost:9000 minio minio123
mc-alias-ls:
	mc alias list
mc-make-bucket:
	mc mb play/hm-bucket
mc-cp:
	mc cp README.md play/hm-bucket
minio-console-ui:
	kubectl port-forward service/storage-tiny-console --namespace=tenant-tiny 9443:9443
minio-headless:
	kubectl port-forward service/storage-tiny-hl --namespace=tenant-tiny 9000:9000

# PostgreSQL
postgres-connect:
	psql postgresql://admin@localhost:5432/postgres
migrate-up:
	migrate \
		-database "postgresql://admin:passw0rd@localhost:5432/opa_db?sslmode=disable&search_path=public" \
		-path data/postgres/opa_db/migrations up
migrate-down:
	migrate \
		-database "postgresql://admin:passw0rd@localhost:5432/opa_db?sslmode=disable&search_path=public" \
		-path data/postgres/opa_db/migrations down

# TorchServe
torchserve-init:
	conda install pytorch torchvision torchaudio --channel=pytorch --yes
	conda install torchserve torch-model-archiver torch-workflow-archiver --channel=pytorch --yes
	pip install captum
torchserve-start:
	torchserve --start --model-store=data/model-server/model-store --models=densenet161=densenet161.mar
torchserve-stop:
	torchserve --stop
curl-list-models:
	curl http://localhost:8081/models
curl-download-cat:
	curl https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg > kitten_small.jpg
curl-cat:
	curl http://127.0.0.1:8080/predictions/densenet161 --upload-file kitten_small.jpg

# Cloudflare
cloudflared-install:
	brew install cloudflared
cloudflare-tunnel-login:
	cloudflared tunnel login
cloudflare-tunnel-list:
	cloudflared tunnel list
cloudflare-tunnel-create:
	cloudflared tunnel create hm-tunnel
cloudflare-tunnel-delete:
	cloudflared tunnel delete hm-tunnel
cloudflare-k8s-create-secret:
	kubectl create secret generic tunnel-credentials \
		--from-file=credentials.json=/Users/hongbo-miao/.cloudflared/c9aa4140-fee8-4862-a479-3c1faacbd816.json \
		--namespace=hm-cloudflared
cloudflare-tunnel-dns:
	cloudflared tunnel route dns hm-tunnel prefect.hongbomiao.com
