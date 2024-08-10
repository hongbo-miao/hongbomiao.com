argo-cd-app-create:
	argocd app create hm-goldilocks --file=argo-cd-manifests/hm-goldilocks-application.yaml
argo-cd-app-update:
	argocd app create hm-goldilocks --file=argo-cd-manifests/hm-goldilocks-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-goldilocks --yes
