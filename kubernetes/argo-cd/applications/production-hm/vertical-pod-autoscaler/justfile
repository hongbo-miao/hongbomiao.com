argo-cd-app-create:
	argocd app create production-hm-vertical-pod-autoscaler --file=argo-cd-manifests/hm-vertical-pod-autoscaler-application.yaml
argo-cd-app-update:
	argocd app create production-hm-vertical-pod-autoscaler --file=argo-cd-manifests/hm-vertical-pod-autoscaler-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete production-hm-vertical-pod-autoscaler --yes
