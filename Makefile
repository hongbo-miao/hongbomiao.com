# Docker
d-build:
	docker build --file Dockerfile.api-go.development --tag api-go-dev .

d-build-no-cache:
	docker build --file Dockerfile.api-go.development --tag api-go-dev --no-cache .

d-run-api-go-dev:
	docker run -p 5000:5000 api-go-dev

d-run-api-go:
	docker run -p 5000:5000 api-go

d-run-api-dev:
	docker run -p 5000:5000 --env-file ./api/.env.development.local.example.docker api-dev

d-run-api:
	docker run -p 5000:5000 --env-file ./api/.env.production.local.example api

d-sh:
	docker run -it api-go sh

d-ps:
	docker ps

d-prune:
	docker system prune

# Docker Compose
dc-build:
	docker-compose --file docker-compose.development.yml build

dc-up:
	docker-compose --file docker-compose.development.yml up --detach

dc-stop:
	docker-compose --file docker-compose.development.yml stop

dc-down:
	docker-compose --file docker-compose.development.yml down --volumes

# Kubernetes
k8s-start:
	minikube start

k8s-ingress:
	minikube addons enable ingress

k8s-apply:
	kubectl apply -f kubernetes

k8s-pods:
	kubectl get pods

k8s-services:
	kubectl get services

k8s-namespaces:
	kubectl get namespaces

k8s-dashboard:
	minikube dashboard

k8s-ip:
	minikube ip

k8s-dev:
	skaffold dev

k8s-delete:
	minikube delete

# Prometheus
prom-curl:
	curl http://localhost:9464/metrics

prom-test:
	docker build --file Dockerfile.prometheus.test .
