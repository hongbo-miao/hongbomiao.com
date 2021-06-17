# Docker
d-build:
	docker build --file=web/Dockerfile --tag=web .
	docker build --file=api/Dockerfile.development --tag=api-dev .
	docker build --file=api/Dockerfile --tag=api .
	docker build --file=Dockerfile.api-go.development --tag=api-go-dev .
	docker build --file=Dockerfile.api-go.production --tag=api-go .
	docker build --file=Dockerfile.api-go-grpc.development --tag=api-go-grpc-dev .
	docker build --file=Dockerfile.api-go-grpc.production --tag=api-go-grpc .

d-build-no-cache:
	docker build --file=Dockerfile.api-go.development --tag=api-go-dev --no-cache .

d-run:
	docker run -p 80:80 web
	docker run -p 5000:5000 --env-file=./api/.env.development.local.example.docker api-dev
	docker run -p 5000:5000 --env-file=./api/.env.production.local.example api
	docker run -p 5000:5000 api-go-dev
	docker run -p 5000:5000 --env=APP_ENV=production api-go
	docker run -p 5000:5000 api-go-grpc-dev
	docker run -p 5000:5000 --env=APP_ENV=production --env=GRPC_HOST=0.0.0.0 api-go-grpc

d-sh:
	docker run -it api-go sh

d-ps:
	docker ps

d-prune:
	docker system prune

# Docker Compose
dc-build:
	docker-compose --file=docker-compose.development.yml build
	docker-compose --file=docker-compose.cypress.yml build

dc-up:
	docker-compose --file=docker-compose.development.yml up --detach
	docker-compose --file=docker-compose.cypress.yml up --detach

dc-stop:
	docker-compose --file=docker-compose.development.yml stop
	docker-compose --file=docker-compose.cypress.yml stop

dc-down:
	docker-compose --file=docker-compose.development.yml down --volumes
	docker-compose --file=docker-compose.cypress.yml down --volumes

# Kubernetes
m-start:
	minikube start

m-service-api-go:
	minikube service api-go-service

m-dashboard:
	minikube dashboard

m-delete:
	minikube delete

m-ip:
	minikube ip

k-apply:
	kubectl apply -f kubernetes

k-delete:
	kubectl delete -f kubernetes

k-pods:
	kubectl get pods

k-services:
	kubectl get services

k-namespaces:
	kubectl get namespaces

k-dev:
	skaffold dev

# Prometheus
prom-curl:
	curl http://localhost:9464/metrics

prom-test:
	docker build --file=Dockerfile.prometheus.test .
