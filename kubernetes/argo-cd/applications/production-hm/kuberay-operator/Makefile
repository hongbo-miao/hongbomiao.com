argo-cd-app-create:
	argocd app create hm-kuberay-operator --file=argo-cd-manifests/hm-kuberay-operator-application.yaml
argo-cd-app-update:
	argocd app create hm-kuberay-operator --file=argo-cd-manifests/hm-kuberay-operator-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-kuberay-operator --yes
