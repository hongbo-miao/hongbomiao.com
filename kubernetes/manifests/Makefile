# Argo CD
download-argocd:
	curl https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml > argocd/argocd.yaml

# Dgraph
download-dgraph:
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-single/dgraph-single.yaml > dgraph/dgraph.yaml
	curl https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml > dgraph/dgraph-ha.yaml

# Fluent Bit
download-fluent-bit:
	curl https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/fluent-bit-service-account.yaml > fluentbit/fluent-bit-service-account.yaml
	curl https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/fluent-bit-role-1.22.yaml > fluentbit/fluent-bit-role-1.22.yaml
	curl https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/fluent-bit-role-binding-1.22.yaml > fluentbit/fluent-bit-role-binding-1.22.yaml
	curl https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/output/elasticsearch/fluent-bit-configmap.yaml > fluentbit/fluent-bit-configmap.yaml
	curl https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/output/elasticsearch/fluent-bit-ds.yaml > fluentbit/fluent-bit-ds.yaml

# Ingress
download-ingress-nginx:
	$(eval INGRESS_VERSION := $(shell curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt))
	curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/$(INGRESS_VERSION)/deploy/static/provider/cloud/deploy.yaml > ingress/ingress-nginx.yaml

# Kafka
download-kafka:
	curl https://strimzi.io/install/latest?namespace=kafka > kafka/strimzi-cluster-operator.yaml
	curl https://strimzi.io/examples/latest/kafka/kafka-persistent-single.yaml > kafka/kafka-persistent-single.yaml

# MinIO
# Manifests: https://github.com/minio/operator/tree/master/examples/kustomization

# Rasa
download-rasa:
	curl https://raw.githubusercontent.com/RasaHQ/helm-charts/main/charts/rasa/values.yaml > rasa/rasa/values.yaml

# Yugabyte
# Docs: https://docs.yugabyte.com/latest/deploy/kubernetes/single-zone/oss/yugabyte-operator/#deploy-a-yugabytedb-cluster-with-this-operator
# Manifests: https://github.com/yugabyte/yugabyte-operator/tree/master/deploy
