download-argocd-yaml:
	curl https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml > argocd.yaml

download-dgraph-yaml:
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-single/dgraph-single.yaml > dgraph.yaml
download-dgraph-ha-yaml:
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml > dgraph-ha.yaml

download-ingress-nginx-yaml:
	$(eval VERSION := $(shell curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt))
	curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/$(VERSION)/deploy/static/provider/cloud/deploy.yaml > ingress-nginx.yaml

download-yugabyte-yaml:
	curl https://raw.githubusercontent.com/yugabyte/yugabyte-db/master/cloud/kubernetes/yugabyte-statefulset-rf-1.yaml > yugabyte-statefulset-rf-1.yaml

# https://docs.yugabyte.com/latest/deploy/kubernetes/single-zone/oss/yugabyte-operator/#deploy-a-yugabytedb-cluster-with-this-operator
# https://github.com/yugabyte/yugabyte-operator/tree/master/deploy
