sealed-secrets-seal:
	cat secrets/production-hm-airbyte-secret.unsealed.yaml | \
	kubeseal \
		--controller-namespace=production-hm-sealed-secrets \
		--controller-name=sealed-secrets \
		--format=yaml \
		> kubernetes-manifests/production-hm-airbyte-secret.yaml

argo-cd-app-create:
	argocd app create production-hm-airbyte --file=production-hm-airbyte.yaml
