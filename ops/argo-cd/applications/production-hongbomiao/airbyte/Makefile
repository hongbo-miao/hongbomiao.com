sealed-secrets-seal:
	cat secrets/hm-airbyte-secret.unsealed.yaml | \
	kubeseal \
		--controller-namespace=production-hm-sealed-secrets \
		--controller-name=hm-sealed-secrets \
		--format=yaml \
		> manifests/hm-airbyte-secret.yaml

argo-cd-app-create:
	argocd app create hm-airbyte --file=manifests/hm-airbyte.yaml
