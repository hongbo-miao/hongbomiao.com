argo-cd-password-reset:
	argocd account bcrypt --password=xxx
	# $2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	kubectl patch secret argocd-secret \
		--namespace=production-hm-argo-cd \
		--patch='{"stringData": {"admin.password": "$2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "admin.passwordMtime": "'$(date +%FT%T%Z)'"}}'
argo-cd-sign-in:
	argocd login localhost:31026 --username=admin --insecure
argo-cd-app-create:
	argocd app create hm-sealed-secrets --file=argo-cd/manifests/hm-sealed-secrets.yaml
argo-cd-app-sync:
	argocd app sync hm-sealed-secrets
