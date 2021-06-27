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
	minikube service api-go-service --namespace=hm
minikube-dashboard:
	minikube dashboard
minikube-ip:
	minikube ip

# Kubernetes
kubectl-apply:
	kubectl apply -f kubernetes/*-namespace.yaml
	kubectl apply -f kubernetes
kubectl-apply-with-linkerd:
	linkerd inject - | kubectl apply -f kubernetes
kubectl-delete:
	kubectl delete -f kubernetes/*.yaml
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

# Argo CD
argocd-install:
	kubectl create namespace argocd
	kubectl apply --namespace=argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
argocd-check:
	for deploy in "dex-server" "redis" "repo-server" "server"; \
	  do kubectl --namespace=argocd rollout status deploy/argocd-$${deploy}; \
	done
argocd-ui:
	kubectl port-forward svc/argocd-server -n argocd 8080:443
argocd-get-password:
	kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo
argocd-login:
	argocd login localhost:8080
argocd-sync:
	argocd app sync hm-application
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
