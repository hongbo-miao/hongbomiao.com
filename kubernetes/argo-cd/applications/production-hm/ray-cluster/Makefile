argo-cd-app-create:
	argocd app create hm-ray-cluster --file=argo-cd-manifests/hm-ray-cluster-application.yaml
argo-cd-app-update:
	argocd app create hm-ray-cluster --file=argo-cd-manifests/hm-ray-cluster-application.yaml --upsert
argo-cd-app-delete:
	argocd app delete hm-ray-cluster --yes

# https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/prometheus-grafana.html
fetch-grafana-dashboards:
	kubectl cp production-hm-ray-cluster/hm-ray-cluster-kuberay-head-xxxxx:/tmp/ray/session_latest/metrics/grafana/dashboards/ /tmp/ray
