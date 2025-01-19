argo-cd-app-create:
	argocd app create production-hm-argo-cd --file=argo-cd-manifests/hm-argo-cd-application.yaml
argo-cd-app-update:
	argocd app create production-hm-argo-cd --file=argo-cd-manifests/hm-argo-cd-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-argo-cd --yes
