sealed-secrets-seal:
	cat secrets/production-hm-airbyte-secret.unsealed.yaml | \
	kubeseal \
		--controller-namespace=production-hm-sealed-secrets \
		--controller-name=sealed-secrets \
		--format=yaml \
		> manifests/production-hm-airbyte-secret.yaml

kubectl-apply:
	kubectl apply --filename=manifests

argo-cd-app-create:
	argocd app create production-hm-airbyte --file=production-hm-airbyte.yaml
	argocd app sync production-hm-airbyte
