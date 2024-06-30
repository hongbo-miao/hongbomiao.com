sealed-secrets-seal:
	cat secrets/hm-airbyte-secret.unsealed.yaml | \
	kubeseal \
		--controller-namespace=production-hm-sealed-secrets \
		--controller-name=hm-sealed-secrets \
		--format=yaml \
		> kubernetes-manifests/hm-airbyte-secret.yaml

argo-cd-app-create:
	argocd app create hm-airbyte --file=argo-cd-manifests/hm-airbyte.yaml
argo-cd-app-update:
	argocd app create hm-airbyte --file=argo-cd-manifests/hm-airbyte.yaml --upsert
