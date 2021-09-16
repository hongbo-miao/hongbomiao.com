# Ingress
download-ingress-nginx-yaml:
	$(eval INGRESS_VERSION := $(shell curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt))
	curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/$(INGRESS_VERSION)/deploy/static/provider/cloud/deploy.yaml > ingress/ingress-nginx.yaml

# Argo CD
download-argocd-yaml:
	curl https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml > argocd/argocd.yaml

# Dgraph
download-dgraph-yaml:
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-single/dgraph-single.yaml > dgraph/dgraph.yaml
download-dgraph-ha-yaml:
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml > dgraph/dgraph-ha.yaml

# Yugabyte
# Docs: https://docs.yugabyte.com/latest/deploy/kubernetes/single-zone/oss/yugabyte-operator/#deploy-a-yugabytedb-cluster-with-this-operator
# Manifests: https://github.com/yugabyte/yugabyte-operator/tree/master/deploy

# MinIO
# Manifests: https://github.com/minio/operator/tree/master/examples/kustomization
