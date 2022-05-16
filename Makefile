# bin
setup:
	sh kubernetes/bin/setup.sh
clean:
	sh kubernetes/bin/clean.sh
setup-k3d:
	sh kubernetes/bin/setupK3d.sh
setup-kind:
	sh kubernetes/bin/setupKind.sh
setup-minikube:
	sh kubernetes/bin/setupMinikube.sh

focus-opa-debug:
	sh kubernetes/bin/utils/focusOPADebug.sh
focus-pixie-debug:
	sh kubernetes/bin/utils/focusPixieDebug.sh
focus-rasa-debug:
	sh kubernetes/bin/utils/focusRasaDebug.sh

install-alpine:
	sh kubernetes/bin/utils/installAlpine.sh

setup-local:
	sh bin/setup.sh
build-local:
	sh bin/build.sh
clean-local:
	sh bin/clean.sh

# Kubernetes
k8s-debug:
	nslookup graphql-server-service.hm.svc.cluster.local

# Docker
docker-login:
	docker login
docker-build:
	docker build --file=api-node/Dockerfile.development --tag=ghcr.io/hongbo-miao/hm-api-node-dev .
	docker build --file=api-node/Dockerfile --tag=ghcr.io/hongbo-miao/hm-api-node .
	docker build --file=api-go/build/package/graphql_server/Dockerfile --tag=ghcr.io/hongbo-miao/hm-graphql-server .
docker-run:
	docker run -p 5000:5000 --name=hm_api_node_dev --rm --env-file=./api/.env.development.local.example.docker hm-api-node-dev
	docker run -p 5000:5000 --name=hm_api_node --rm --env-file=./api/.env.production.local.example hm-api-node
	docker run -p 31800:31800 --name=hm_graphql_server --rm --env=APP_ENV=production hm-graphql-server
docker-sh:
	docker run --rm -it hm-graphql-server sh
docker-ps:
	docker ps
docker-ps-all:
	docker ps --all
docker-rmi:
	docker rmi --force IMAGE_ID
docker-prune:
	docker system prune

# Docker Compose
docker-compose-build:
	docker-compose --file=docker-compose.development.yml build
	docker-compose --file=docker-compose.cypress.yml build
docker-compose-up:
	docker-compose --file=docker-compose.development.yml up --detach
	docker-compose --file=docker-compose.cypress.yml up --detach
docker-compose-stop:
	docker-compose --file=docker-compose.development.yml stop
	docker-compose --file=docker-compose.cypress.yml stop
docker-compose-down:
	docker-compose --file=docker-compose.development.yml down --volumes
	docker-compose --file=docker-compose.cypress.yml down --volumes

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
	minikube mount ./kubernetes/data:/data
minikube-delete:
	minikube delete
minikube-dashboard:
	minikube dashboard
minikube-ip:
	minikube ip

# kind
kind-create:
	kind create cluster --name=west --config=kubernetes/kind/west-cluster-config.yaml
	kind create cluster --name=east --config=kubernetes/kind/east-cluster-config.yaml
kind-delete:
	kind delete cluster --name=west
	kind delete cluster --name=east
kind-list-clusters:
	kind get clusters

# k3d
k3d-cluster-create:
	k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
	k3d cluster create east --config=kubernetes/k3d/west-cluster-config.yaml
	k3d cluster create dev --config=kubernetes/k3d/dev-cluster-config.yaml
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
	multipass mount "${HOME}/Clouds/Git/hongbomiao.com/kubernetes/data" west-master:/data
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

# Kubernetes
kubectl-apply:
	kubectl apply --filename=kubernetes/manifests/west/hm-namespace.yaml
	kubectl apply --filename=kubernetes/manifests/west
	kubectl apply --filename=kubernetes/manifests/west/hm-limitrange.yaml
	kubectl apply --filename=kubernetes/manifests/west --selector=app.kubernetes.io/name=graphql-server
kubectl-apply-with-linkerd:
	linkerd inject - | kubectl apply --filename=kubernetes/manifests
kubectl-delete:
	kubectl delete --filename=kubernetes/manifests
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
kubectl-context-info:
	kubectl cluster-info --context k3d-west

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
	# opa:
	kubectl port-forward service/graphql-server-service --namespace=hm 8181:8181
	# opal-client
	kubectl port-forward service/graphql-server-service --namespace=hm 7000:7000
kubectl-port-forward-opal-server:
	kubectl port-forward service/opal-server-service --namespace=hm 7002:7002
kubectl-port-forward-config-server:
	kubectl port-forward service/config-server-service --namespace=hm 26660:26660

kubectl-port-forward-postgres:
	kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072
kubectl-port-forward-timescale:
	kubectl port-forward service/timescale --namespace=hm-timescale 25495:25495
kubectl-port-forward-redis:
	kubectl port-forward service/redis-headless --namespace=hm-redis 6379:6379
kubectl-port-forward-trino:
	kubectl port-forward service/trino --namespace=hm-trino 26869:26869
kubectl-port-forward-superset:
	kubectl port-forward service/superset --namespace=hm-superset 22357:22357
kubectl-port-forward-dgraph:
	kubectl port-forward service/dgraph-dgraph-alpha --namespace=hm-dgraph 8080:8080
	kubectl port-forward service/dgraph-dgraph-zero --namespace=hm-dgraph 6080:6080
kubectl-port-forward-elastic:
	kubectl port-forward service/hm-kibana-kb-http --namespace=elastic 5601:5601
	kubectl port-forward service/hm-elasticsearch-es-http --namespace=elastic 9200:9200
kubectl-port-forward-kafka:
	kubectl port-forward service/hm-kafkaconnect-connect-api --namespace=kafka 8083:8083
kubectl-port-forward-flink:
	kubectl port-forward service/hm-flink-cluster-rest --namespace=hm-flink 8081:8081
kubectl-port-forward-prometheus:
	kubectl port-forward service/monitoring-kube-prometheus-prometheus --namespace=hm-monitoring 9090:9090
kubectl-port-forward-grafana:
	kubectl port-forward service/monitoring-grafana --namespace=hm-monitoring 54976:80

kubectl-port-forward-ory-hydra:
	kubectl port-forward service/ory-hydra-public --namespace=hm-ory-hydra 4444:4444
	kubectl port-forward service/ory-hydra-admin --namespace=hm-ory-hydra 4445:4445
kubectl-port-forward-rasa:
	kubectl port-forward service/hm-release-rasa --namespace=hm-rasa 39143:5005
	kubectl port-forward service/hm-release-postgresql-headless --namespace=hm-rasa 35785:5432
	kubectl port-forward service/hm-release-redis-master --namespace=hm-rasa 6379:6379
	kubectl port-forward service/hm-release-rasa-action-server --namespace=hm-rasa-action-server 5055:5055
kubectl-port-forward-goldilocks:
	kubectl port-forward service/goldilocks-dashboard --namespace=hm-goldilocks 34617:34617
kubectl-port-forward-kubecost:
	kubectl port-forward service/kubecost-cost-analyzer --namespace=hm-kubecost 9090:9090
kubectl-port-forward-polaris:
	kubectl port-forward service/polaris-dashboard --namespace=hm-polaris 31450:80
kubectl-port-forward-sloop:
	kubectl port-forward service/sloop --namespace=hm-sloop 16850:16850
kubectl-port-forward-rancher:
	kubectl port-forward service/rancher --namespace=cattle-system 46271:443
kubectl-port-forward-argocd:
	kubectl port-forward service/argocd-server --namespace=argocd 31026:443

list-port-forward:
	ps -ef | grep port-forward
kill-port:
	kill -9 PID

# Vagrant
vagrant-init:
	vagrant init hashicorp/bionic64
vagrant-up:
	vagrant up
vagrant-suspend:
	vagrant suspend
vagrant-halt:
	vagrant halt
vagrant-destroy:
	vagrant destroy
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
vagrant-box-remove:
	vagrant box remove hashicorp/bionic64
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
	linkerd install --values=kubernetes/linkerd/config.yaml --disable-heartbeat | kubectl apply --filename=-
linkerd-install-control-plane-prod:
	linkerd install --values=kubernetes/linkerd/config.yaml --disable-heartbeat --ha | kubectl apply --filename=-
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
	kubectl port-forward service/hm-kb-http --namespace=elastic 5601:5601 &
kibana-get-password: # username: elastic
	kubectl get secret hm-elasticsearch-es-elastic-user --namespace=elastic --output=jsonpath="{.data.elastic}" | base64 --decode; echo

# Argo CD
argocd-install:
	kubectl create namespace argocd
	kubectl apply --namespace=argocd --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
argocd-ui:
	kubectl port-forward service/argocd-server --namespace=argocd 31026:443 &
argocd-get-password: # username: admin
	kubectl get secret argocd-initial-admin-secret --namespace=argocd --output=jsonpath="{.data.password}" | base64 -d && echo
argocd-login:
	$(eval ARGOCD_PASSWORD := $(shell kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo))
	argocd login localhost:31026 --username=admin --password=$(ARGOCD_PASSWORD) --insecure
argocd-enable-auth-sync:
	argocd app set hm-application --sync-policy=automated
argocd-disable-auth-sync:
	argocd app set hm-application --sync-policy=none
argocd-diff:
	argocd app diff hm-application --local=kubernetes/manifests
argocd-apply:
	kubectl apply --filename=kubernetes/manifests/argocd/hm-application.yaml
argocd-sync:
	kubectl apply --filename=kubernetes/manifests/argocd/hm-application.yaml
	argocd app sync hm-application --grpc-web
argocd-sync-local:
	argocd app sync hm-application --grpc-web --local=kubernetes/manifests/west
argocd-sync-local-full:
	kubectl create namespace hm
	ELASTIC_APM_TOKEN=$(kubectl get secret hm-apm-apm-token --namespace=elastic --output=go-template='{{index .data "secret-token" | base64decode}}')
	kubectl create secret generic hm-elastic-apm --namespace=hm --from-literal="token=${ELASTIC_APM_TOKEN}"
	kubectl apply --filename=kubernetes/manifests/argocd/hm-application.yaml
	argocd app sync hm-application --grpc-web --local=kubernetes/manifests/west
argocd-list:
	argocd app list
argocd-delete:
	argocd app delete hm-application --yes
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
	docker run -it -p=5080:5080 -p=6080:6080 -p 8080:8080 \
      -p=9080:9080 -p=8000:8000 -v=~/dgraph:/dgraph --name=dgraph \
      dgraph/standalone
dgraph-install:
	kubectl apply --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-single/dgraph-single.yaml
dgraph-delete:
	kubectl delete --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-single/dgraph-single.yaml
	kubectl delete persistentvolumeclaims --namespace=hm --selector=app=dgraph
dgraph-install-ha:
	kubectl apply --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml
dgraph-delete-ha:
	kubectl delete --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml
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
	psql postgresql://admin@localhost:40072/postgres
migrate-up:
	$(eval POSTGRESQL_URL := "postgresql://admin:passw0rd@localhost:40072/opa_db?sslmode=disable&search_path=public")
	migrate -database "${POSTGRESQL_URL}" -path kubernetes/data/postgres/opa_db/migrations up
migrate-down:
	$(eval POSTGRESQL_URL := "postgresql://admin:passw0rd@localhost:40072/opa_db?sslmode=disable&search_path=public")
	migrate -database "${POSTGRESQL_URL}" -path kubernetes/data/postgres/opa_db/migrations down

# Kafka
zookeeper-start:
	zookeeper-server-start /usr/local/etc/kafka/zookeeper.properties
kafka-start:
	kafka-server-start /usr/local/etc/kafka/server.properties

kafka-topic-create:
	kafka-topics --bootstrap-server=localhost:9092 --create --topic=my-topic --partitions=3 --replication-factor=1
kafka-topic-delete:
	kafka-topics --bootstrap-server=localhost:9092 --delete --topic=my-topic
kafka-topic-list:
	kafka-topics --bootstrap-server=localhost:9092 --list
kafka-topic-describe:
	kafka-topics --bootstrap-server=localhost:9092 --topic=my-topic --describe

kafka-console-producer:
	kafka-console-producer --bootstrap-server=localhost:9092 --topic=my-topic
kafka-console-consumer:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic
kafka-console-consumer-from-beginning:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --from-beginning
kafka-console-consumer-group:
	kafka-console-consumer --bootstrap-server=localhost:9092 --topic=my-topic --group=my-group

kafka-consumer-group-list:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --list
kafka-consumer-group-describe:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --describe --group=my-group
kafka-consumer-group-reset-offset-to-earliest:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --group=my-group --topic=my-topic --reset-offsets --to-earliest --execute
kafka-consumer-group-reset-offset-shift-by:
	kafka-consumer-groups --bootstrap-server=localhost:9092 --group=my-group --topic=my-topic --reset-offsets --shift-by=-1 --execute

# Prometheus
prom-curl:
	curl http://localhost:9464/metrics
prom-test:
	docker build --file=Dockerfile.prometheus.test .

# TorchServe
torchserve-init:
	conda install pytorch torchvision torchaudio --channel=pytorch
	conda install torchserve torch-model-archiver torch-workflow-archiver --channel=pytorch
	pip install captum
torchserve-start:
	torchserve --start --model-store=kubernetes/data/model-server/model-store --models=densenet161=densenet161.mar
torchserve-stop:
	torchserve --stop
curl-list-models:
	curl http://localhost:8081/models
curl-download-cat:
	curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
curl-cat:
	curl http://127.0.0.1:8080/predictions/densenet161 --upload-file kitten_small.jpg

# Cloudflare
cloudflare-tunnel-login:
	cloudflared tunnel login
cloudflare-tunnel-list:
	cloudflared tunnel list
cloudflare-tunnel-create:
	cloudflared tunnel create hm-tunnel
cloudflare-tunnel-delete:
	cloudflared tunnel delete hm-tunnel
cloudflare-k8s-create-secret:
	kubectl create secret generic tunnel-credentials --from-file=credentials.json=/Users/homiao/.cloudflared/afccc94c-0065-4e40-832a-a00b1940faaf.json
cloudflare-tunnel-dns:
	cloudflared tunnel route dns hm-tunnel tunnel.hongbomiao.com

# Python
python-static-type-check:
	poetry run poe mypy convolutional-neural-network --install-types --non-interactive
	poetry run poe mypy graph-neural-network --install-types --non-interactive
	poetry run poe mypy locust --install-types --non-interactive
	poetry run poe mypy hm-opal-client --install-types --non-interactive
lint-python-black:
	poetry run poe lint-py-black
lint-python-black-fix:
	poetry run poe lint-py-black-fix
lint-python-flake8:
	poetry run poe lint-py-flake8
lint-python-isort:
	poetry run poe lint-py-isort
lint-python-isort-fix:
	poetry run poe lint-py-isort-fix

# Lint
lint-dockerfile:
	hadolint $$(git ls-files "**/Dockerfile*")
lint-shell:
	shellcheck $$(git ls-files "**/*.sh")
lint-kubernetes:
	kubeconform \
		-kubernetes-version=1.23.4 \
		-ignore-filename-pattern=".*trafficsplit.yaml" \
		-ignore-filename-pattern=".*my-values.yaml" \
		-ignore-filename-pattern="kubernetes/manifests/argocd/" \
		-ignore-filename-pattern="kubernetes/manifests/elastic/" \
		-ignore-filename-pattern="kubernetes/manifests/kafka/" \
		-ignore-filename-pattern="kubernetes/manifests/prometheus/" \
		-ignore-filename-pattern="kubernetes/manifests/yugabyte/" \
		$$(git ls-files "kubernetes/manifests/")
lint-protocol-buffers:
	buf lint
