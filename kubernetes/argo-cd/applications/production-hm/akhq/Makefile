argo-cd-app-create:
	argocd app create hm-akhq --file=argo-cd-manifests/hm-akhq-application.yaml
argo-cd-app-update:
	argocd app create hm-akhq --file=argo-cd-manifests/hm-akhq-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-akhq --yes
