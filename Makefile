# Docker
docker-build:
	docker-compose --file docker-compose.development.yml build

docker-up:
	docker-compose --file docker-compose.development.yml up --detach

docker-stop:
	docker-compose --file docker-compose.development.yml stop

docker-down:
	docker-compose --file docker-compose.development.yml down --volumes

docker-delete:
	docker system prune

# Kubernetes
k8s-start:
	minikube start

k8s-ingress:
	minikube addons enable ingress

k8s-apply:
	kubectl apply --file kubernetes

k8s-pods:
	kubectl get pods

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
