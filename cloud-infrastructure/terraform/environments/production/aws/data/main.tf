# Amazon S3 bucket - hm-production-bucket
module "production_hm_production_bucket_amazon_s3_bucket" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-production-bucket"
  environment    = var.environment
  team           = var.team
}

# Amazon EKS
locals {
  amazon_eks_cluster_name = "hm-production-eks-cluster"
}
module "hm_amazon_eks_access_entry_iam" {
  providers                    = { aws = aws.production }
  source                       = "../../../../modules/aws/hm_amazon_eks_access_entry_iam"
  amazon_eks_access_entry_name = "hm-production-eks-cluster-access-entry"
  environment                  = var.environment
  team                         = var.team
}
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest
module "eks" {
  source                         = "terraform-aws-modules/eks/aws"
  version                        = "20.13.1"
  cluster_name                   = local.amazon_eks_cluster_name
  cluster_version                = "1.30"
  cluster_endpoint_public_access = true
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
  }
  vpc_id                   = "vpc-xxxxxxxxxxxxxxxxx"
  subnet_ids               = ["subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx"]
  control_plane_subnet_ids = ["subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx"]
  eks_managed_node_group_defaults = {
    instance_types = ["m7i.large", "m7g.large", "m6i.large", "m6in.large", "m5.large", "m5n.large", "m5zn.large"]
  }
  eks_managed_node_groups = {
    hm_node_group = {
      min_size       = 2
      max_size       = 10
      desired_size   = 2
      instance_types = ["m7i.large", "m6i.large", "m6in.large", "m5.large", "m5n.large", "m5zn.large"]
      capacity_type  = "SPOT"
    }
  }
  enable_cluster_creator_admin_permissions = true
  access_entries = {
    hm_access_entry = {
      kubernetes_groups = []
      principal_arn     = module.hm_amazon_eks_access_entry_iam.arn
      policy_associations = {
        hm_policy_association = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSViewPolicy"
          access_scope = {
            namespaces = ["default"]
            type       = "namespace"
          }
        }
      }
    }
  }
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = local.amazon_eks_cluster_name
  }
}
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest/submodules/karpenter
module "karpenter" {
  source       = "terraform-aws-modules/eks/aws//modules/karpenter"
  cluster_name = module.eks.cluster_name
  # Attach additional IAM policies to the Karpenter node IAM role
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.amazon_eks_cluster_name}-karpenter"
  }
}

# Argo CD
module "hm_kubernetes_namespace_hm_argo_cd" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-argo-cd"
  depends_on = [
    module.eks
  ]
}
module "hm_argo_cd_helm_chart" {
  source = "../../../../modules/kubernetes/hm_argo_cd_helm_chart"
  name   = "hm-argo-cd"
  # https://artifacthub.io/packages/helm/argo/argo-cd
  argo_cd_version = "7.1.3"
  namespace       = module.hm_kubernetes_namespace_hm_argo_cd.namespace
  environment     = var.environment
  team            = var.team
  depends_on = [
    module.hm_kubernetes_namespace_hm_argo_cd
  ]
}

# Sealed Secrets
module "hm_kubernetes_namespace_hm_sealed_secrets" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-sealed-secrets"
  depends_on = [
    module.eks
  ]
}

# Monitoring
module "hm_kubernetes_namespace_hm_monitoring" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-monitoring"
  depends_on = [
    module.eks
  ]
}

