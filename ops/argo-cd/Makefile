argo-cd-password-reset:
	argocd account bcrypt --password=xxx
	# $2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	kubectl patch secret argocd-secret \
		--namespace=production-hm-argo-cd \
		--patch='{"stringData": {"admin.password": "$2a$10$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "admin.passwordMtime": "'$(date +%FT%T%Z)'"}}'

argo-cd-sign-in:
	argocd login localhost:31026 --username=admin --insecure

argo-cd-project-create:
	argocd proj create hongbomiao --file
argo-cd-project-delete:
	argocd proj delete hongbomiao
argo-cd-project-list:
	argocd proj list

argo-cd-repo-add:
	argocd repo add git@github.com:hongbo-miao/hongbomiao.com.git --ssh-private-key-path=$$HOME/.ssh/hm_argo_cd_id_ed25519 --project=hongbomiao
argo-cd-repo-list:
	argocd repo list
argo-cd-repo-remove:
	argocd repo rm git@github.com:hongbo-miao/hongbomiao.com.git
