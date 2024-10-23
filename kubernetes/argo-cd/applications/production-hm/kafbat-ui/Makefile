argo-cd-app-create:
	argocd app create production-hm-kafbat-ui --file=argo-cd-manifests/hm-kafbat-ui-application.yaml
argo-cd-app-update:
	argocd app create production-hm-kafbat-ui --file=argo-cd-manifests/hm-kafbat-ui-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-kafbat-ui --yes
