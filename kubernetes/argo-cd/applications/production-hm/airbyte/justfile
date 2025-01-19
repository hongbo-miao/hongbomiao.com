sealed-secrets-seal:
	cat secrets/hm-airbyte-secret.unsealed.yaml | \
	kubeseal \
		--controller-namespace=production-hm-sealed-secrets \
		--controller-name=hm-sealed-secrets \
		--format=yaml \
		> kubernetes-manifests/hm-airbyte-secret.yaml

argo-cd-app-create:
	argocd app create production-hm-airbyte --file=argo-cd-manifests/hm-airbyte-application.yaml
argo-cd-app-update:
	argocd app create production-hm-airbyte --file=argo-cd-manifests/hm-airbyte-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-airbyte --yes
