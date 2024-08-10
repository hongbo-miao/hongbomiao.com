argo-cd-app-create:
	argocd app create hm-mlflow --file=argo-cd-manifests/hm-mlflow-application.yaml
argo-cd-app-update:
	argocd app create hm-mlflow --file=argo-cd-manifests/hm-mlflow-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-mlflow --yes
