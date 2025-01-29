data "terraform_remote_state" "production_aws_network_terraform_remote_state" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
}

# Amazon EKS
locals {
  amazon_eks_cluster_name      = "hm-kubernetes"
  amazon_eks_cluster_name_hash = substr(md5(local.amazon_eks_cluster_name), 0, 3)
}
# Amazon EKS Access Entry - IAM role
module "amazon_eks_access_entry_iam" {
  providers                    = { aws = aws.production }
  source                       = "../../../../modules/kubernetes/hm_amazon_eks_access_entry_iam_role"
  amazon_eks_access_entry_name = local.amazon_eks_cluster_name
  amazon_eks_cluster_name_hash = local.amazon_eks_cluster_name_hash
  environment                  = var.environment
  team                         = var.team
}
# Amazon EBS CSI Driver - IAM role
module "amazon_ebs_csi_driver_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_amazon_ebs_csi_driver_iam_role"
  amazon_eks_cluster_name              = module.amazon_eks_cluster.cluster_name
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  amazon_eks_cluster_name_hash         = local.amazon_eks_cluster_name_hash
  environment                          = var.environment
  team                                 = var.team
}
# Amazon S3 CSI driver mountpoint - S3 bucket
module "s3_bucket_eks_cluster_mount" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.amazon_eks_cluster_name}-mount"
  environment    = var.environment
  team           = var.team
}
# Amazon S3 CSI driver mountpoint - IAM role
module "amazon_s3_csi_driver_mountpoint_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_amazon_s3_csi_driver_mountpoint_iam_role"
  amazon_eks_cluster_name              = module.amazon_eks_cluster.cluster_name
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  s3_bucket_name                       = module.amazon_s3_bucket_eks_cluster_mount.name
  environment                          = var.environment
  team                                 = var.team
}
# Amazon EKS cluster
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest
module "amazon_eks_cluster" {
  providers       = { aws = aws.production }
  source          = "terraform-aws-modules/eks/aws"
  version         = "20.33.1"
  cluster_name    = local.amazon_eks_cluster_name
  cluster_version = "1.30"
  cluster_addons = {
    coredns = {
      addon_version               = "v1.11.1-eksbuild.9"
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
    kube-proxy = {
      addon_version               = "v1.30.0-eksbuild.3"
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
    vpc-cni = {
      addon_version               = "v1.18.3-eksbuild.1"
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
    aws-ebs-csi-driver = {
      addon_version               = "v1.33.0-eksbuild.1"
      service_account_role_arn    = module.amazon_ebs_csi_driver_iam_role.arn
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
    aws-mountpoint-s3-csi-driver = {
      addon_version               = "v1.7.0-eksbuild.1"
      service_account_role_arn    = module.amazon_s3_csi_driver_mountpoint_iam_role.arn
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
  }
  cluster_endpoint_public_access = false
  cluster_service_ipv4_cidr      = "10.215.0.0/16"
  cluster_security_group_additional_rules = {
    ingress_rule_on_site = {
      description = "On-Site"
      type        = "ingress"
      cidr_blocks = ["10.10.0.0/15"]
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
    },
    ingress_rule_vpn = {
      description = "VPN"
      type        = "ingress"
      cidr_blocks = ["10.100.0.0/15"]
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
    },
    ingress_rule_vpc = {
      description = "VPC"
      type        = "ingress"
      cidr_blocks = [data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_ipv4_cidr_block]
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
    }
  }
  vpc_id                   = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id
  subnet_ids               = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids
  control_plane_subnet_ids = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids
  eks_managed_node_group_defaults = {
    instance_types = ["m7i.large", "m7g.large", "m6i.large", "m6in.large", "m5.large", "m5n.large", "m5zn.large"]
  }
  eks_managed_node_groups = {
    hm_eks_node_group = {
      min_size       = 10
      max_size       = 50
      desired_size   = 10
      instance_types = ["m7a.2xlarge", "m7i.2xlarge", "m6a.2xlarge", "m6i.2xlarge", "m6in.2xlarge", "m5.2xlarge", "m5a.2xlarge", "m5n.2xlarge", "m5zn.2xlarge"]
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
      principal_arn     = module.amazon_eks_access_entry_iam.arn
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
    Environment  = var.environment
    Team         = var.team
    ResourceName = local.amazon_eks_cluster_name
  }
  depends_on = [
    module.amazon_eks_access_entry_iam
  ]
}

# Karpenter
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest/submodules/karpenter
module "karpenter" {
  source       = "terraform-aws-modules/eks/aws//modules/karpenter"
  cluster_name = module.amazon_eks_cluster.cluster_name
  # Attach additional IAM policies to the Karpenter node IAM role
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "${local.amazon_eks_cluster_name}-karpenter"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Gateway API
# Gateway API - S3 bucket
module "s3_bucket_eks_cluster_elastic_load_balancer" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.amazon_eks_cluster_name}-elastic-load-balancer"
  environment    = var.environment
  team           = var.team
}
module "s3_bucket_eks_cluster_network_load_balancer_policy" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/kubernetes/hm_aws_network_load_balancer_s3_bucket_policy"
  s3_bucket_name = module.amazon_s3_bucket_eks_cluster_elastic_load_balancer.name
}
# Gateway API - CRDs
module "kubernetes_manifest_gateway_api" {
  source            = "../../../../modules/kubernetes/hm_kubernetes_manifest"
  manifest_dir_path = "files/gateway-api/manifests"
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Argo CD
# Argo CD - Kubernetes namespace
module "kubernetes_namespace_hm_argo_cd" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-argo-cd"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}
# Argo CD - Helm chart
module "argo_cd_helm_chart" {
  source              = "../../../../modules/kubernetes/hm_helm_chart"
  repository          = "https://argoproj.github.io/argo-helm"
  chart_name          = "argo-cd"
  chart_version       = "7.3.5" # https://artifacthub.io/packages/helm/argo/argo-cd
  release_name        = "hm-argo-cd"
  namespace           = module.kubernetes_namespace_hm_argo_cd.namespace
  my_values_yaml_path = "files/argo-cd/helm/my-values.yaml"
  environment         = var.environment
  team                = var.team
  depends_on = [
    module.kubernetes_namespace_hm_argo_cd
  ]
}

# ExternalDNS
# ExternalDNS - IAM role
module "external_dns_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_external_dns_iam_role"
  external_dns_service_account_name    = "hm-external-dns"
  external_dns_namespace               = "${var.environment}-hm-external-dns"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  amazon_route53_hosted_zone_id        = var.amazon_route53_hosted_zone_id
  amazon_eks_cluster_name_hash         = local.amazon_eks_cluster_name_hash
  environment                          = var.environment
  team                                 = var.team
}
# ExternalDNS - Kubernetes namespace
module "kubernetes_namespace_hm_external_dns" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-external-dns"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# cert-manager
# cert-manager - IAM role
module "cert_manager_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_cert_manager_iam_role"
  cert_manager_service_account_name    = "hm-cert-manager"
  cert_manager_namespace               = "${var.environment}-hm-cert-manager"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  amazon_route53_hosted_zone_id        = var.amazon_route53_hosted_zone_id
  amazon_eks_cluster_name_hash         = local.amazon_eks_cluster_name_hash
  environment                          = var.environment
  team                                 = var.team
}
# cert-manager - Kubernetes namespace
module "kubernetes_namespace_hm_cert_manager" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-cert-manager"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Metrics Server
# Metrics Server - Kubernetes namespace
module "kubernetes_namespace_hm_metrics_server" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-metrics-server"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Vertical Pod Autoscaler
# Vertical Pod Autoscaler - Kubernetes namespace
module "kubernetes_namespace_hm_vertical_pod_autoscaler" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-vertical-pod-autoscaler"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Goldilocks (requires Metrics Server and Vertical Pod Autoscaler)
# Goldilocks - Kubernetes namespace
module "kubernetes_namespace_hm_goldilocks" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-goldilocks"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Traefik
# Traefik - Kubernetes namespace
module "kubernetes_namespace_hm_traefik" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-traefik"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Sealed Secrets
# Sealed Secrets - Kubernetes namespace
module "kubernetes_namespace_hm_sealed_secrets" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-sealed-secrets"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Airbyte
# Airbyte - S3 bucket
module "s3_bucket_hm_airbyte" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${var.environment}-hm-airbyte"
  environment    = var.environment
  team           = var.team
}
# Airbyte - IAM user
module "airbyte_iam_user" {
  providers         = { aws = aws.production }
  source            = "../../../../modules/kubernetes/hm_airbyte_iam_user"
  aws_iam_user_name = "${var.environment}_hm_airbyte_user"
  s3_bucket_name    = module.amazon_s3_bucket_hm_airbyte.name
  environment       = var.environment
  team              = var.team
}
# Airbyte - Postgres
locals {
  airbyte_postgres_name = "${var.environment}-hm-airbyte-postgres"
}
data "aws_secretsmanager_secret" "hm_airbyte_postgres_secret" {
  provider = aws.production
  name     = "${var.environment}-hm-airbyte-postgres/admin"
}
data "aws_secretsmanager_secret_version" "hm_airbyte_postgres_secret_version" {
  provider  = aws.production
  secret_id = data.aws_secretsmanager_secret.hm_airbyte_postgres_secret.id
}
module "airbyte_postgres_security_group" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_amazon_rds_security_group"
  amazon_ec2_security_group_name = "${local.airbyte_postgres_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id
  amazon_vpc_cidr_ipv4           = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_ipv4_cidr_block
  environment                    = var.environment
  team                           = var.team
}
module "airbyte_postgres_subnet_group" {
  providers         = { aws = aws.production }
  source            = "../../../../modules/aws/hm_amazon_rds_subnet_group"
  subnet_group_name = "${local.airbyte_postgres_name}-subnet-group"
  subnet_ids        = var.amazon_vpc_private_subnet_ids
  environment       = var.environment
  team              = var.team
}
module "airbyte_postgres_parameter_group" {
  providers            = { aws = aws.production }
  source               = "../../../../modules/aws/hm_amazon_rds_parameter_group"
  family               = "postgres16"
  parameter_group_name = "${local.airbyte_postgres_name}-parameter-group"
  # https://github.com/airbytehq/airbyte/issues/39636
  parameters = [
    {
      name  = "rds.force_ssl"
      value = "0"
    }
  ]
  environment = var.environment
  team        = var.team
}
module "airbyte_postgres_instance" {
  providers                  = { aws = aws.production }
  source                     = "../../../../modules/aws/hm_amazon_rds_instance"
  amazon_rds_name            = local.airbyte_postgres_name
  amazon_rds_engine          = "postgres"
  amazon_rds_engine_version  = "16.3"
  amazon_rds_instance_class  = "db.m7g.large"
  amazon_rds_storage_size_gb = 32
  user_name                  = jsondecode(data.aws_secretsmanager_secret_version.hm_airbyte_postgres_secret_version.secret_string)["user_name"]
  password                   = jsondecode(data.aws_secretsmanager_secret_version.hm_airbyte_postgres_secret_version.secret_string)["password"]
  parameter_group_name       = module.airbyte_postgres_parameter_group.name
  subnet_group_name          = module.airbyte_postgres_subnet_group.name
  vpc_security_group_ids     = [module.airbyte_postgres_security_group.id]
  cloudwatch_log_types       = ["postgresql", "upgrade"]
  environment                = var.environment
  team                       = var.team
}
# Airbyte - Kubernetes namespace
module "kubernetes_namespace_hm_airbyte" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-airbyte"
  labels = {
    "goldilocks.fairwinds.com/enabled" = true
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# MLflow
# MLflow - S3 bucket
module "s3_bucket_hm_mlflow" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${var.environment}-hm-mlflow"
  environment    = var.environment
  team           = var.team
}
# MLflow - IAM role
module "mlflow_tracking_server_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_mlflow_tracking_server_iam_role"
  mlflow_service_account_name          = "hm-mlflow-tracking"
  mlflow_namespace                     = "${var.environment}-hm-mlflow"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  s3_bucket_name                       = module.amazon_s3_bucket_hm_mlflow.name
  environment                          = var.environment
  team                                 = var.team
}
module "mlflow_run_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_mlflow_run_iam_role"
  mlflow_service_account_name          = "hm-mlflow-run"
  mlflow_namespace                     = "${var.environment}-hm-mlflow"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  s3_bucket_name                       = module.amazon_s3_bucket_hm_mlflow.name
  environment                          = var.environment
  team                                 = var.team
}
# MLflow - Postgres
locals {
  mlflow_postgres_name = "${var.environment}-hm-mlflow-postgres"
}
data "aws_secretsmanager_secret" "hm_mlflow_postgres_secret" {
  provider = aws.production
  name     = "${var.environment}-hm-mlflow-postgres/admin"
}
data "aws_secretsmanager_secret_version" "hm_mlflow_postgres_secret_version" {
  provider  = aws.production
  secret_id = data.aws_secretsmanager_secret.hm_mlflow_postgres_secret.id
}
module "mlflow_postgres_security_group" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_amazon_rds_security_group"
  amazon_ec2_security_group_name = "${local.mlflow_postgres_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id
  amazon_vpc_cidr_ipv4           = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_ipv4_cidr_block
  environment                    = var.environment
  team                           = var.team
}
module "mlflow_postgres_subnet_group" {
  providers         = { aws = aws.production }
  source            = "../../../../modules/aws/hm_amazon_rds_subnet_group"
  subnet_group_name = "${local.mlflow_postgres_name}-subnet-group"
  subnet_ids        = var.amazon_vpc_private_subnet_ids
  environment       = var.environment
  team              = var.team
}
module "mlflow_postgres_parameter_group" {
  providers            = { aws = aws.production }
  source               = "../../../../modules/aws/hm_amazon_rds_parameter_group"
  family               = "postgres16"
  parameter_group_name = "${local.mlflow_postgres_name}-parameter-group"
  environment          = var.environment
  team                 = var.team
}
module "mlflow_postgres_instance" {
  providers                  = { aws = aws.production }
  source                     = "../../../../modules/aws/hm_amazon_rds_instance"
  amazon_rds_name            = local.mlflow_postgres_name
  amazon_rds_engine          = "postgres"
  amazon_rds_engine_version  = "16.3"
  amazon_rds_instance_class  = "db.m7g.large"
  amazon_rds_storage_size_gb = 32
  user_name                  = jsondecode(data.aws_secretsmanager_secret_version.hm_mlflow_postgres_secret_version.secret_string)["user_name"]
  password                   = jsondecode(data.aws_secretsmanager_secret_version.hm_mlflow_postgres_secret_version.secret_string)["password"]
  parameter_group_name       = module.mlflow_postgres_parameter_group.name
  subnet_group_name          = module.mlflow_postgres_subnet_group.name
  vpc_security_group_ids     = [module.mlflow_postgres_security_group.id]
  cloudwatch_log_types       = ["postgresql", "upgrade"]
  environment                = var.environment
  team                       = var.team
}
# MLflow - Kubernetes namespace
module "kubernetes_namespace_hm_mlflow" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-mlflow"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Ray
# KubeRay - Kubernetes namespace
module "kubernetes_namespace_hm_kuberay_operator" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-kuberay-operator"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}
# Ray Cluster - IAM role
module "ray_cluster_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_ray_cluster_iam_role"
  ray_cluster_service_account_name     = "hm-ray-cluster-service-account"
  ray_cluster_namespace                = "${var.environment}-hm-ray-cluster"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  s3_bucket_name                       = module.amazon_s3_bucket_hm_mlflow.name
  environment                          = var.environment
  team                                 = var.team
}
# Ray Cluster - Kubernetes namespace
module "kubernetes_namespace_hm_ray_cluster" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-ray-cluster"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Prometheus
# Prometheus - Kubernetes namespace
module "kubernetes_namespace_hm_prometheus" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-prometheus"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Loki
# Loki - S3 bucket
locals {
  loki_admin_name = "${var.environment}-hm-loki-admin"
  loki_chunk_name = "${var.environment}-hm-loki-chunk"
  loki_ruler_name = "${var.environment}-hm-loki-ruler"
}
module "s3_bucket_hm_loki_admin" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.loki_admin_name}-bucket"
  environment    = var.environment
  team           = var.team
}
module "s3_bucket_hm_loki_chunk" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.loki_chunk_name}-bucket"
  environment    = var.environment
  team           = var.team
}
module "s3_bucket_hm_loki_ruler" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.loki_ruler_name}-bucket"
  environment    = var.environment
  team           = var.team
}
module "hm_loki_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_loki_iam_role"
  loki_service_account_name            = "hm-loki"
  loki_namespace                       = "${var.environment}-hm-loki"
  loki_admin_s3_bucket_name            = module.s3_bucket_hm_loki_admin.name
  loki_chunk_s3_bucket_name            = module.s3_bucket_hm_loki_chunk.name
  loki_ruler_s3_bucket_name            = module.s3_bucket_hm_loki_ruler.name
  amazon_eks_cluster_oidc_provider     = module.hm_amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.hm_amazon_eks_cluster.oidc_provider_arn
  environment                          = var.environment
  team                                 = var.team
}
# Loki - Kubernetes namespace
module "kubernetes_namespace_hm_loki" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-loki"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Promtail
# Promtail - Kubernetes namespace
module "kubernetes_namespace_hm_promtail" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-promtail"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Alloy
# Alloy - Kubernetes namespace
module "kubernetes_namespace_hm_alloy" {
  source               = "../../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-alloy"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Tempo
# Tempo - S3 bucket
locals {
  tempo_admin_name = "${var.environment}-hm-tempo-admin"
  tempo_trace_name = "${var.environment}-hm-tempo-trace"
}
module "s3_bucket_hm_tempo_admin" {
  providers      = { aws = aws.engineering }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.tempo_admin_name}-bucket"
  environment    = var.environment
  team           = var.team
}
module "s3_bucket_hm_tempo_trace" {
  providers      = { aws = aws.engineering }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.tempo_trace_name}-bucket"
  environment    = var.environment
  team           = var.team
}
module "hm_tempo_iam_role" {
  providers                            = { aws = aws.engineering }
  source                               = "../../../../modules/kubernetes/hm_tempo_iam_role"
  tempo_service_account_name           = "hm-tempo"
  tempo_namespace                      = "${var.environment}-hm-tempo"
  tempo_admin_s3_bucket_name           = module.s3_bucket_hm_tempo_admin.name
  tempo_trace_s3_bucket_name           = module.s3_bucket_hm_tempo_trace.name
  amazon_eks_cluster_oidc_provider     = module.hm_amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.hm_amazon_eks_cluster.oidc_provider_arn
  environment                          = var.environment
  team                                 = var.team
}
# Tempo - Kubernetes namespace
module "kubernetes_namespace_hm_tempo" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-tempo"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Grafana
# Grafana - Kubernetes namespace
module "kubernetes_namespace_hm_grafana" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-grafana"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Qdrant
# Qdrant - Kubernetes namespace
module "kubernetes_namespace_hm_qdrant" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-qdrant"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Valkey
# Valkey - Kubernetes namespace
module "kubernetes_namespace_hm_valkey" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-valkey"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Netdata
# Netdata - Kubernetes namespace
module "kubernetes_namespace_hm_netdata" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-netdata"
  labels = {
    "goldilocks.fairwinds.com/enabled" = true
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# OpenCost
# OpenCost - Kubernetes namespace
module "kubernetes_namespace_hm_opencost" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-opencost"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Confluent Schema Registry
# Confluent Schema Registry - IAM role
module "confluent_schema_registry_iam_role" {
  providers                                          = { aws = aws.production }
  source                                             = "../../../../modules/kubernetes/hm_confluent_schema_registry_iam_role"
  confluent_schema_registry_service_account_nickname = "hm-schema-registry-service-account"
  confluent_schema_registry_service_account_name     = "hm-confluent-schema-registry-service-account"
  confluent_schema_registry_namespace                = "production-hm-confluent-schema-registry"
  amazon_eks_cluster_oidc_provider                   = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn               = module.amazon_eks_cluster.oidc_provider_arn
  amazon_msk_arn                                     = "arn:aws:kafka:us-west-2:272394222652:cluster/production-iot-kafka/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-xx"
  environment                                        = var.environment
  team                                               = var.team
}
# Confluent Schema Registry - Kubernetes namespace
module "kubernetes_namespace_hm_confluent_schema_registry" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-confluent-schema-registry"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Redpanda Console
# Redpanda Console - IAM role
module "redpanda_console_iam_role" {
  providers                             = { aws = aws.production }
  source                                = "../../../../modules/kubernetes/hm_redpanda_console_iam_role"
  redpanda_console_service_account_name = "hm-redpanda-console"
  redpanda_console_namespace            = "${var.environment}-hm-redpanda-console"
  amazon_eks_cluster_oidc_provider      = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn  = module.amazon_eks_cluster.oidc_provider_arn
  environment                           = var.environment
  team                                  = var.team
}
# Redpanda Console - Kubernetes namespace
module "kubernetes_namespace_hm_redpanda_console" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-redpanda-console"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Kafbat UI
# Kafbat UI - IAM role
module "kafbat_ui_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_kafbat_ui_iam_role"
  kafbat_ui_service_account_name       = "hm-kafbat-ui"
  kafbat_ui_namespace                  = "${var.environment}-hm-kafbat-ui"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  environment                          = var.environment
  team                                 = var.team
}
# Kafbat UI - Kubernetes namespace
module "kubernetes_namespace_hm_kafbat_ui" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-kafbat-ui"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# LiteLLM
# LiteLLM - IAM role
module "hm_litellm_iam_role" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/kubernetes/hm_litellm_iam_role"
  litellm_service_account_name         = "hm-litellm-service-account"
  litellm_namespace                    = "${var.environment}-hm-litellm"
  amazon_eks_cluster_oidc_provider     = module.amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.amazon_eks_cluster.oidc_provider_arn
  environment                          = var.environment
  team                                 = var.team
}
# LiteLLM - Kubernetes namespace
module "kubernetes_namespace_hm_litellm" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-litellm"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}

# Open WebUI
# Open WebUI - Kubernetes namespace
module "kubernetes_namespace_hm_open_webui" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-open-webui"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}
# Open WebUI Pipelines - Kubernetes namespace
module "kubernetes_namespace_hm_open_webui_pipelines" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-open-webui-pipelines"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.amazon_eks_cluster
  ]
}
