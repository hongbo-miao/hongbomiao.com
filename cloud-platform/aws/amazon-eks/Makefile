amazon-eks-describe:
	aws eks describe-cluster --name=hm-eks-cluster

amazon-eks-upgrade-kubernetes-version:
	eksctl upgrade nodegroup \
		--name=hm-k8s-cluster-node-group \
		--cluster=hm-k8s-cluster \
		--region=us-west-2 \
		--kubernetes-version=1.29
