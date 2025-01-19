argo-cd-app-create:
	argocd app create production-hm-confluent-schema-registry --file=argo-cd-manifests/hm-confluent-schema-registry-application.yaml
argo-cd-app-update:
	argocd app create production-hm-confluent-schema-registry --file=argo-cd-manifests/hm-confluent-schema-registry-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-confluent-schema-registry --yes
