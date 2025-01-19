sealed-secrets-seal:
	cat secrets/hm-open-webui-secret.unsealed.yaml | \
	kubeseal \
		--controller-namespace=production-hm-sealed-secrets \
		--controller-name=hm-sealed-secrets \
		--format=yaml \
		> kubernetes-manifests/hm-open-webui-secret.yaml

argo-cd-app-create:
	argocd app create production-hm-open-webui --file=argo-cd-manifests/hm-open-webui-application.yaml
argo-cd-app-update:
	argocd app create production-hm-open-webui --file=argo-cd-manifests/hm-open-webui-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-open-webui --yes
