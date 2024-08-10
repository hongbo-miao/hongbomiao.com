argo-cd-app-create:
	argocd app create hm-redpanda-console --file=manifests/hm-redpanda-console-application.yaml
argo-cd-app-update:
	argocd app create hm-redpanda-console --file=manifests/hm-redpanda-console-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-redpanda-console --yes
