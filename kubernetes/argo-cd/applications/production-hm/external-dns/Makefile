argo-cd-app-create:
	argocd app create production-hm-external-dns --file=argo-cd-manifests/hm-external-dns-application.yaml
argo-cd-app-update:
	argocd app create production-hm-external-dns --file=argo-cd-manifests/hm-external-dns-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-external-dns --yes
