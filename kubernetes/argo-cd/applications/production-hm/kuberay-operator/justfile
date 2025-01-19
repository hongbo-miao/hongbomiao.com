argo-cd-app-create:
	argocd app create production-hm-kuberay-operator --file=argo-cd-manifests/hm-kuberay-operator-application.yaml
argo-cd-app-update:
	argocd app create production-hm-kuberay-operator --file=argo-cd-manifests/hm-kuberay-operator-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-kuberay-operator --yes
