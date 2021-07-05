# bin
setup:
	sh bin/setup.sh
build:
	sh bin/build.sh
clean:
	sh bin/clean.sh
setup-kubernetes:
	sh bin/setup-kubernetes.sh

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

# minikube
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
	minikube service web-service --namespace=hm
minikube-service-api-go:
	kubectl port-forward service/api-go-service --namespace=hm 5000:5000
minikube-service-opa:
	kubectl port-forward service/api-go-service --namespace=hm 8181:8181
minikube-service-opal-server:
	kubectl port-forward service/opal-server --namespace=hm 7002:7002

minikube-dashboard:
	minikube dashboard
minikube-ip:
	minikube ip

# Kubernetes
kubectl-apply:
	kubectl apply --filename=kubernetes/hm-namespace.yaml
	kubectl apply --filename=kubernetes
kubectl-apply-with-linkerd:
	linkerd inject - | kubectl apply --filename=kubernetes
kubectl-delete:
	kubectl delete --filename=kubernetes
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
	linkerd install --disable-heartbeat | kubectl apply --filename=-
linkerd-install-control-plane-prod:
	linkerd install --disable-heartbeat --ha | kubectl apply --filename=-
linkerd-install-viz:
	linkerd viz install | kubectl apply --filename=-
linkerd-install-jaeger:
	linkerd jaeger install | kubectl apply --filename=-
linkerd-get-yaml:
	linkerd install --disable-heartbeat > linkerd.yaml
linkerd-viz-dashboard:
	linkerd viz dashboard
linkerd-jaeger-dashboard:
	linkerd jaeger dashboard
linkerd-inject:
	kubectl get deployments --namespace=hm --output=yaml | linkerd inject - | kubectl apply --filename=-
linkerd-check-pre:
	linkerd check --pre
linkerd-check:
	linkerd check
linkerd-viz-tap:
	linkerd viz tap deployments/api-go-deployment --namespace=hm
linkerd-viz-tap-verbose:
	linkerd viz tap deployments/api-go-deployment --namespace=hm --output=json
linkerd-viz-tap-filter:
	linkerd viz tap deployments/api-go-deployment --namespace=hm --to=deployment/api-go-grpc-deployment
	linkerd viz tap deployments/api-go-deployment --namespace=hm --to=deployment/api-go-grpc-deployment --path=/api.proto.greet.v1.GreetService/Greet
linkerd-viz-top: # shows traffic routes sorted by the most popular paths
	linkerd viz top deployments/api-go-deployment --namespace=hm

# Argo CD
argocd-install:
	kubectl create namespace argocd
	kubectl apply --namespace=argocd --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
argocd-check:
	for deploy in "dex-server" "redis" "repo-server" "server"; \
	  do kubectl --namespace=argocd rollout status deployments/argocd-$${deploy}; \
	done
argocd-ui:
	kubectl port-forward service/argocd-server --namespace=argocd 8080:443
argocd-get-password: # username: admin
	kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo
argocd-login:
	argocd login localhost:8080
argocd-enable-auth-sync:
	argocd app set hm-application --sync-policy=automated
argocd-disable-auth-sync:
	argocd app set hm-application --sync-policy=none
argocd-diff:
	argocd app diff hm-application --local=kubernetes
argocd-sync:
	argocd app sync hm-application
argocd-sync-local:
	argocd app sync hm-application --local=kubernetes
argocd-list:
	argocd app list

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

# Lint
hadolint:
	hadolint $$(git ls-files '**/Dockerfile*')
shellcheck:
	shellcheck $$(git ls-files '**/*.sh')
kubeval:
	kubeval $$(git ls-files 'kubernetes/*.yaml')
