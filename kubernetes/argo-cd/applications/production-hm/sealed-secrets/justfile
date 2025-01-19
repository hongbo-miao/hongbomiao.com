argo-cd-app-create:
	argocd app create production-hm-sealed-secrets --file=argo-cd-manifests/hm-sealed-secrets-application.yaml
argo-cd-app-update:
	argocd app create production-hm-sealed-secrets --file=argo-cd-manifests/hm-sealed-secrets-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-sealed-secrets --yes
