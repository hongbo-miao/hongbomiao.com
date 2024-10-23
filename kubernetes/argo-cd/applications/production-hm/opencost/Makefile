argo-cd-app-create:
	argocd app create production-hm-opencost --file=argo-cd-manifests/hm-opencost-application.yaml
argo-cd-app-update:
	argocd app create production-hm-opencost --file=argo-cd-manifests/hm-opencost-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-opencost --yes
