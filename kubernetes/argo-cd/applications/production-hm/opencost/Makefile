argo-cd-app-create:
	argocd app create hm-opencost --file=argo-cd-manifests/hm-opencost-application.yaml
argo-cd-app-update:
	argocd app create hm-opencost --file=argo-cd-manifests/hm-opencost-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-opencost --yes
