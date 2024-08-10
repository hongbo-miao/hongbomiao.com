argo-cd-app-create:
	argocd app create hm-confluent-schema-registry --file=argo-cd-manifests/hm-confluent-schema-registry-application.yaml
argo-cd-app-update:
	argocd app create hm-confluent-schema-registry --file=argo-cd-manifests/hm-confluent-schema-registry-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-confluent-schema-registry --yes
