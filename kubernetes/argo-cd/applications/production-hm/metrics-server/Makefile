argo-cd-app-create:
	argocd app create hm-metrics-server --file=argo-cd-manifests/hm-metrics-server-application.yaml
argo-cd-app-update:
	argocd app create hm-metrics-server --file=argo-cd-manifests/hm-metrics-server-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-metrics-server --yes
