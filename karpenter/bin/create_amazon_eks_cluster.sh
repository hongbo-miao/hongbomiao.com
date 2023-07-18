#!/usr/bin/env bash
set -e

# https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/

echo "# Set environment variables"
export KARPENTER_VERSION=v0.29.1
export AWS_PARTITION="aws"
export CLUSTER_NAME="hm-k8s-cluster"
export AWS_DEFAULT_REGION="us-west-2"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query=Account --output=text)"
export AWS_ACCOUNT_ID

TEMPOUT=$(mktemp)
export TEMPOUT
echo "=================================================="

echo "# Create a cluster"
curl --silent --fail --show-error --location "https://raw.githubusercontent.com/aws/karpenter/${KARPENTER_VERSION}/website/content/en/preview/getting-started/getting-started-with-karpenter/cloudformation.yaml" > "${TEMPOUT}" && \
aws cloudformation deploy \
  --stack-name="${CLUSTER_NAME}-karpenter-stack" \
  --template-file="${TEMPOUT}" \
  --capabilities=CAPABILITY_NAMED_IAM \
  --parameter-overrides="ClusterName=${CLUSTER_NAME}"

eksctl create cluster --config-file=- <<EOF
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_DEFAULT_REGION}
  version: "1.27"
  tags:
    karpenter.sh/discovery: ${CLUSTER_NAME}
iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: karpenter
        namespace: karpenter
      roleName: ${CLUSTER_NAME}-karpenter
      attachPolicyARNs:
        - arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerPolicy-${CLUSTER_NAME}
      roleOnly: true
iamIdentityMappings:
  - arn: "arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:role/KarpenterNodeRole-${CLUSTER_NAME}"
    username: system:node:{{EC2PrivateDNSName}}
    groups:
      - system:bootstrappers
      - system:nodes
managedNodeGroups:
  - instanceType: m6a.xlarge
    amiFamily: AmazonLinux2
    name: ${CLUSTER_NAME}-node-group
    desiredCapacity: 2
    minSize: 1
    maxSize: 100
EOF
CLUSTER_ENDPOINT="$(aws eks describe-cluster --name=${CLUSTER_NAME} --query="cluster.endpoint" --output=text)"
export CLUSTER_ENDPOINT
echo "${CLUSTER_ENDPOINT}"

KARPENTER_IAM_ROLE_ARN="arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER_NAME}-karpenter"
export CLUSTER_ENDPOINT
echo "${KARPENTER_IAM_ROLE_ARN}"
echo "=================================================="

echo "# Verify the cluster"
aws iam create-service-linked-role --aws-service-name spot.amazonaws.com || true
# If the role has already been successfully created, you will see:
# An error occurred (InvalidInput) when calling the CreateServiceLinkedRole operation: Service role name AWSServiceRoleForEC2Spot has been taken in this account, please try a different suffix.
echo "=================================================="

echo "# Install Karpenter"
# Logout of helm registry to perform an unauthenticated pull against the public ECR
helm registry logout public.ecr.aws
helm upgrade \
  karpenter \
  oci://public.ecr.aws/karpenter/karpenter \
  --install \
  --namespace=karpenter \
  --create-namespace \
  --version="${KARPENTER_VERSION}" \
  --set="serviceAccount.annotations.eks\.amazonaws\.com/role-arn=${KARPENTER_IAM_ROLE_ARN}" \
  --set="settings.aws.clusterName=${CLUSTER_NAME}" \
  --set="settings.aws.defaultInstanceProfile=KarpenterNodeInstanceProfile-${CLUSTER_NAME}" \
  --set="settings.aws.interruptionQueueName=${CLUSTER_NAME}" \
  --set=controller.resources.requests.cpu=1 \
  --set=controller.resources.requests.memory=1Gi \
  --set=controller.resources.limits.cpu=1 \
  --set=controller.resources.limits.memory=1Gi \
  --wait
echo "=================================================="

echo "# Create provisioner"
cat <<EOF | kubectl apply --filename=-
---
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: default
spec:
  requirements:
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["spot"]
  limits:
    resources:
      cpu: 1000
  providerRef:
    name: default
  consolidation:
    enabled: true
---
apiVersion: karpenter.k8s.aws/v1alpha1
kind: AWSNodeTemplate
metadata:
  name: default
spec:
  subnetSelector:
    karpenter.sh/discovery: ${CLUSTER_NAME}
  securityGroupSelector:
    karpenter.sh/discovery: ${CLUSTER_NAME}
EOF
echo "=================================================="

echo "# Install Prometheus"
kubectl create namespace monitoring

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/prometheus \
  --namespace=monitoring \
  --values=kubernetes/manifests-raw/karpenter/prometheus-values.yaml
echo "=================================================="

echo "# Install Grafana"
helm repo add grafana-charts https://grafana.github.io/helm-charts
helm repo update
helm install grafana grafana-charts/grafana \
  --namespace=monitoring \
  --values=kubernetes/manifests-raw/karpenter/grafana-values.yaml

# kubectl port-forward service/grafana --namespace=monitoring 45767:80

# Username: admin
# Password:
#   kubectl get secret grafana \
#     --namespace=monitoring \
#     --output=jsonpath="{.data.admin-password}" \
#     | base64 --decode
echo "=================================================="

echo "# Cleanup"
helm uninstall karpenter --namespace=karpenter
aws cloudformation delete-stack --stack-name="${CLUSTER_NAME}-karpenter-stack"
aws ec2 describe-launch-templates --filters="Name=tag:karpenter.k8s.aws/cluster,Values=${CLUSTER_NAME}" |
    jq -r ".LaunchTemplates[].LaunchTemplateName" |
    xargs -I{} aws ec2 delete-launch-template --launch-template-name {}
eksctl delete cluster --name="${CLUSTER_NAME}"
echo "=================================================="
