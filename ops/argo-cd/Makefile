argo-cd-password-reset:
	argocd account bcrypt --password=xxx
	# $2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	kubectl patch secret argocd-secret \
		--namespace=horizon-production-argo-cd \
		--patch='{"stringData": {"admin.password": "$2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "admin.passwordMtime": "'$(date +%FT%T%Z)'"}}'
