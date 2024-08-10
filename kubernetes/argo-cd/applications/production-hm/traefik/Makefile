argo-cd-app-create:
	argocd app create hm-traefik --file=argo-cd-manifests/hm-traefik-application.yaml
argo-cd-app-update:
	argocd app create hm-traefik --file=argo-cd-manifests/hm-traefik-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-traefik --yes
