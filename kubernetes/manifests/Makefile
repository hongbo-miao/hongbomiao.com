download-argocd-yaml:
	curl https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml > argocd.yaml

download-dgraph-single-yaml:
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-single/dgraph-single.yaml > dgraph-single.yaml

download-ingress-nginx-yaml:
	$(eval VERSION := $(shell curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt))
	curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/$(VERSION)/deploy/static/provider/cloud/deploy.yaml > ingress-nginx.yaml
