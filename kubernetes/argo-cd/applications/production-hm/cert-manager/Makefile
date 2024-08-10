argo-cd-app-create:
	argocd app create hm-cert-manager --file=argo-cd-manifests/hm-cert-manager-application.yaml
argo-cd-app-update:
	argocd app create hm-cert-manager --file=argo-cd-manifests/hm-cert-manager-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-cert-manager --yes
