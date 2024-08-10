argo-cd-app-create:
	argocd app create hm-netdata --file=argo-cd-manifests/hm-netdata-application.yaml
argo-cd-app-update:
	argocd app create hm-netdata --file=argo-cd-manifests/hm-netdata-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-netdata --yes
