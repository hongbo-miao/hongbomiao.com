argo-cd-app-create:
	argocd app create hm-caddy-ingress-controller --file=argo-cd-manifests/hm-caddy-ingress-controller-application.yaml
argo-cd-app-update:
	argocd app create hm-caddy-ingress-controller --file=argo-cd-manifests/hm-caddy-ingress-controller-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-caddy-ingress-controller --yes
