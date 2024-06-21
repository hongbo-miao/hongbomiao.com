data "aws_vpc" "hm_amazon_vpc" {
  default = true
}

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
  version                        = "20.14.0"
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
      min_size       = 10
      max_size       = 50
      desired_size   = 10
      instance_types = ["m7i.large", "m6i.large", "m6in.large", "m5.large", "m5n.large", "m5zn.large"]
      capacity_type  = "SPOT"
    }
  }
  node_security_group_additional_rules = {
    # For kubeseal
    ingress_8080 = {
      type                          = "ingress"
      protocol                      = "tcp"
      from_port                     = 8080
      to_port                       = 8080
      source_cluster_security_group = true
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
  argo_cd_version     = "7.2.0"
  namespace           = module.hm_kubernetes_namespace_hm_argo_cd.namespace
  my_values_yaml_path = "files/argo-cd/helm/my-values.yaml"
  environment         = var.environment
  team                = var.team
  depends_on = [
    module.hm_kubernetes_namespace_hm_argo_cd
  ]
}

# Airbyte
# Airbyte - S3 bucket
module "hm_amazon_s3_bucket_hm_airbyte" {
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${var.environment}-hm-airbyte"
  environment    = var.environment
  team           = var.team
}
# Airbyte - IAM user
module "hm_airbyte_iam_user" {
  source            = "../../../../modules/aws/hm_airbyte_iam_user"
  aws_iam_user_name = "${var.environment}_hm_airbyte_user"
  s3_bucket_name    = module.hm_amazon_s3_bucket_hm_airbyte.name
  environment       = var.environment
  team              = var.team
}
# Airbyte - Postgres
locals {
  amazon_rds_name = "${var.environment}-hm-airbyte-postgres"
}
data "aws_secretsmanager_secret" "hm_airbyte_postgres_secret" {
  name = "${var.environment}-hm-airbyte-postgres/admin"
}
data "aws_secretsmanager_secret_version" "hm_airbyte_postgres_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_airbyte_postgres_secret.id
}
module "hm_hm_airbyte_postgres_security_group" {
  source                         = "../../../../modules/aws/hm_airbyte_postgres_security_group"
  amazon_ec2_security_group_name = "${local.amazon_rds_name}-security-group"
  amazon_vpc_id                  = data.aws_vpc.hm_amazon_vpc.id
  environment                    = var.environment
  team                           = var.team
}
module "hm_hm_airbyte_postgres_subnet_group" {
  source            = "../../../../modules/aws/hm_amazon_rds_subnet_group"
  subnet_group_name = "${local.amazon_rds_name}-subnet-group"
  subnet_ids        = ["subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx", "subnet-xxxxxxxxxxxxxxxxx"]
  environment       = var.environment
  team              = var.team
}
module "hm_hm_airbyte_postgres_parameter_group" {
  source               = "../../../../modules/aws/hm_amazon_rds_parameter_group"
  family               = "postgres16"
  parameter_group_name = "${local.amazon_rds_name}-parameter-group"
  # https://stackoverflow.com/questions/78645095/pod-airbyte-temporal-failed-to-connect-to-rds-with-rds-force-ssl-enabled
  parameters = [
    {
      name  = "rds.force_ssl"
      value = "0"
    }
  ]
  environment = var.environment
  team        = var.team
}
module "hm_hm_airbyte_postgres_instance" {
  source                     = "../../../../modules/aws/hm_amazon_rds_instance"
  amazon_rds_name            = local.amazon_rds_name
  amazon_rds_engine          = "postgres"
  amazon_rds_engine_version  = "16.3"
  amazon_rds_instance_class  = "db.m7g.large"
  amazon_rds_storage_size_gb = 32
  user_name                  = jsondecode(data.aws_secretsmanager_secret_version.hm_airbyte_postgres_secret_version.secret_string)["user_name"]
  password                   = jsondecode(data.aws_secretsmanager_secret_version.hm_airbyte_postgres_secret_version.secret_string)["password"]
  parameter_group_name       = module.hm_hm_airbyte_postgres_parameter_group.name
  subnet_group_name          = module.hm_hm_airbyte_postgres_subnet_group.name
  vpc_security_group_ids     = [module.hm_hm_airbyte_postgres_security_group.id]
  cloudwatch_log_types       = ["postgresql", "upgrade"]
  environment                = var.environment
  team                       = var.team
}
module "hm_kubernetes_namespace_hm_airbyte" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-airbyte"
  depends_on = [
    module.eks
  ]
}

# Metrics Server
module "hm_kubernetes_namespace_hm_metrics_server" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-metrics-server"
  depends_on = [
    module.eks
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

# Prometheus
module "hm_kubernetes_namespace_hm_prometheus" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-prometheus"
  depends_on = [
    module.eks
  ]
}
