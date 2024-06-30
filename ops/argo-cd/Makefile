# Argo CD
argo-cd-password-reset:
	argocd account bcrypt --password=xxx
	# $2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	kubectl patch secret argocd-secret \
		--namespace=production-hm-argo-cd \
		--patch='{"stringData": {"admin.password": "$2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "admin.passwordMtime": "'$(date +%FT%T%Z)'"}}'

argo-cd-sign-in:
	argocd login localhost:31026 --username=admin --insecure

argo-cd-project-create:
	argocd proj create production-hm --file
argo-cd-project-delete:
	argocd proj delete production-hm
argo-cd-project-list:
	argocd proj list

argo-cd-repo-add:
	argocd repo add git@github.com:hongbo-miao/hongbomiao.com.git --ssh-private-key-path=$$HOME/.ssh/hm_argo_cd_id_ed25519 --project=hongbomiao
argo-cd-repo-list:
	argocd repo list
argo-cd-repo-remove:
	argocd repo rm git@github.com:hongbo-miao/hongbomiao.com.git

argo-cd-app-create:
	argocd app create xxx --file=xxx.yaml
argo-cd-app-sync:
	argocd app sync xxx

# Kubernetes
kubectl-port-forward-airbyte:
	kubectl port-forward service/horizon-airbyte-airbyte-webapp-svc --namespace=production-hm-airbyte 28811:443
kubectl-port-forward-argo-cd:
	kubectl port-forward service/hm-argo-cd-argocd-server --namespace=production-hm-argo-cd 31026:443
kubectl-port-forward-goldilocks:
	kubectl port-forward service/hm-goldilocks-dashboard --namespace=production-hm-goldilocks 34617:80
kubectl-port-forward-grafana:
	kubectl port-forward service/hm-prometheus-grafana --namespace=production-hm-prometheus 54976:80
kubectl-port-forward-opencost:
	kubectl port-forward service/hm-opencost --namespace=production-hm-opencost 56053:9090
