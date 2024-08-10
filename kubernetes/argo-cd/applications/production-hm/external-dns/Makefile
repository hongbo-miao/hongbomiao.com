argo-cd-app-create:
	argocd app create hm-external-dns --file=argo-cd-manifests/hm-external-dns-application.yaml
argo-cd-app-update:
	argocd app create hm-external-dns --file=argo-cd-manifests/hm-external-dns-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-external-dns --yes
