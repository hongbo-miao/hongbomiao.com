argo-cd-app-create:
	argocd app create hm-vertical-pod-autoscaler --file=argo-cd-manifests/hm-vertical-pod-autoscaler-application.yaml
argo-cd-app-update:
	argocd app create hm-vertical-pod-autoscaler --file=argo-cd-manifests/hm-vertical-pod-autoscaler-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-vertical-pod-autoscaler --yes
