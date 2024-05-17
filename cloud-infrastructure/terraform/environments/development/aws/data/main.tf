# Amazon S3 bucket - hm-development-bucket
module "development_hm_development_bucket_amazon_s3_bucket" {
  providers      = { aws = aws.development }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-development-bucket"
  environment    = var.environment
  team           = var.team
}

# Amazon EKS
locals {
  amazon_eks_cluster_name = "hm-development-eks-cluster"
}
module "hm_amazon_eks_access_entry_iam" {
  providers                    = { aws = aws.production }
  source                       = "../../../../modules/aws/hm_amazon_eks_access_entry_iam"
  amazon_eks_access_entry_name = "hm-development-eks-cluster-access-entry"
  environment                  = var.environment
  team                         = var.team
}
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest
module "eks" {
  source                         = "terraform-aws-modules/eks/aws"
  version                        = "20.11.0"
  cluster_name                   = local.amazon_eks_cluster_name
  cluster_version                = "1.29"
  cluster_endpoint_public_access = false
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
  # EKS Managed Node Group(s)
  eks_managed_node_group_defaults = {
    instance_types = ["m6i.large", "m6in.large", "m5.large", "m5n.large", "m5zn.large"]
  }
  eks_managed_node_groups = {
    hm_node_group = {
      min_size       = 1
      max_size       = 10
      desired_size   = 1
      instance_types = ["t3.large"]
      capacity_type  = "SPOT"
    }
  }
  # Cluster access entry
  # To add the current caller identity as an administrator
  enable_cluster_creator_admin_permissions = true
  access_entries = {
    # One access entry with a policy associated
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
