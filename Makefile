# bin
setup:
	sh bin/setup.sh

build:
	sh bin/build.sh

clean:
	sh bin/clean.sh

setup-kubernetes:
	sh bin/setup-kubernetes.sh

clean-kubernetes:
	sh bin/clean-kubernetes.sh

# Docker
docker-build:
	docker build --file=web/Dockerfile --tag=hm-web .
	docker build --file=api/Dockerfile.development --tag=hm-api-dev .
	docker build --file=api/Dockerfile --tag=hm-api .
	docker build --file=api-go/Dockerfile.api --tag=hm-api-go .
	docker build --file=api-go/Dockerfile.grpc --tag=hm-api-go-grpc .

docker-run:
	docker run -p 80:80 web
	docker run -p 5000:5000 --name=hm_api_dev --rm --env-file=./api/.env.development.local.example.docker hm-api-dev
	docker run -p 5000:5000 --name=hm_api --rm --env-file=./api/.env.production.local.example hm-api
	docker run -p 5000:5000 --name=hm_api_go --rm --env=APP_ENV=production hm-api-go
	docker run -p 5000:5000 --name=hm_api_go_grpc --rm --env=APP_ENV=production --env=GRPC_HOST=0.0.0.0 hm-api-go-grpc

docker-sh:
	docker run --rm -it hm-api-go sh

docker-ps:
	docker ps
	docker ps --all

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

# Kubernetes
minikube-config:
	minikube config set cpus 4
	minikube config set memory 8192

minikube-start:
	minikube start

minikube-start-hyperkit:
	minikube start --driver=hyperkit

minikube-delete:
	minikube delete

minikube-service-web:
	minikube service --namespace=hm web-service

minikube-service-opal-client:
	minikube service --namespace=hm opal-client

minikube-service-api-go:
	minikube service --namespace=hm api-go-service

minikube-dashboard:
	minikube dashboard

minikube-ip:
	minikube ip

kubectl-apply:
	kubectl apply -f helm-chart/hm-chart/templates/*-namespace.yaml
	kubectl apply -f helm-chart/hm-chart/templates/*.yaml

kubectl-apply-with-linkerd:
	linkerd inject - | kubectl apply -f kubernetes

kubectl-delete:
	kubectl delete -f helm-chart/hm-chart/templates/*.yaml

kubectl-get-pods:
	kubectl get pods --namespace=hm

kubectl-get-services:
	kubectl get services --namespace=hm

kubectl-get-services-yaml:
	kubectl get services api-go-service --namespace=hm --output=yaml

kubectl-get-deployments:
	kubectl get deployments --namespace=hm

kubectl-get-deployments-yaml:
	kubectl get deployments api-go-deployment --namespace=hm --output=yaml

kubectl-get-namespaces:
	kubectl get namespaces

kubectl-get-endpoints:
	kubectl get endpoints api-go-service --namespace=hm

kubectl-get-configmap:
	kubectl get configmap --namespace=hm

kubectl-logs:
	kubectl logs --follow POD_NAME --namespace=hm

kubectl-sh:
	kubectl exec --stdin --tty POD_NAME --namespace=hm -- sh

# Skaffold:
skaffold:
	skaffold dev

# Linkerd
linkerd-install-control-plane:
	linkerd install | kubectl apply -f -

linkerd-install-viz:
	linkerd viz install | kubectl apply -f -

linkerd-install-jaeger:
	linkerd jaeger install | kubectl apply -f -

linkerd-get-yaml:
	linkerd install > linkerd.yaml

linkerd-viz-dashboard:
	linkerd viz dashboard

linkerd-jaeger-dashboard:
	linkerd jaeger dashboard

linkerd-inject:
	kubectl get deployments --namespace=hm --output=yaml | linkerd inject - | kubectl apply -f -

linkerd-check-pre:
	linkerd check --pre

linkerd-check:
	linkerd check

# Helm
helm-install:
	helm install hm-chart helm-chart/hm-chart

helm-install-dry-run:
	helm install hm-chart helm-chart/hm-chart --dry-run

helm-upgrade:
	helm upgrade hm-chart helm-chart/hm-chart

helm-upgrade-dry-run:
	helm upgrade hm-chart helm-chart/hm-chart --dry-run

helm-uninstall:
	helm uninstall hm-chart

helm-uninstall-dry-run:
	helm uninstall hm-chart --dry-run

helm-uninstall-all:
	helm list --all --short | xargs -L1 helm delete

helm-list:
	helm list

# Prometheus
prom-curl:
	curl http://localhost:9464/metrics

prom-test:
	docker build --file=Dockerfile.prometheus.test .

# hadolint
hadolint:
	hadolint $$(git ls-files '**/Dockerfile*')

# shellcheck
shellcheck:
	shellcheck $$(git ls-files '**/*.sh')

# kubeval
kubeval:
	helm template helm-chart/hm-chart | kubeval
