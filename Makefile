# Docker
docker-build:
	docker-compose -f docker-compose.development.yml build

docker-up:
	docker-compose -f docker-compose.development.yml up -d

docker-stop:
	docker-compose -f docker-compose.development.yml stop

docker-down:
	docker-compose -f docker-compose.development.yml down --volumes

docker-delete:
	docker system prune

# Kubernetes
k8s-apply:
	kubectl apply -f k8s

k8s-ip:
	minikube ip

# Prometheus
prometheus:
	curl http://localhost:9464/metrics
