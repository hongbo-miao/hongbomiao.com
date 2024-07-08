data "terraform_remote_state" "hm_terraform_remote_state_production_aws_network" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
}

# Amazon S3 bucket - hm-production-bucket
module "production_hm_production_bucket_amazon_s3_bucket" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-production-bucket"
  environment    = var.environment
  team           = var.team
}

# Tracker Kafka
# Tracker Kafka - S3 bucket
module "hm_amazon_s3_bucket_development_hm_tracker_kafka" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${var.environment}-hm-tracker-kakfa"
  environment    = var.environment
  team           = var.team
}
# Tracker Kafka - Kafka cluster
data "aws_kms_alias" "aws_kms_kafka_key" {
  name = "alias/aws/kafka"
}
locals {
  tracker_kafka_broker_number           = 3
  tracker_amazon_vpc_private_subnet_ids = local.tracker_kafka_broker_number < 4 ? slice(var.amazon_vpc_private_subnet_ids, 0, local.tracker_kafka_broker_number) : var.amazon_vpc_private_subnet_ids
}
module "hm_amazon_msk_cluster" {
  providers                       = { aws = aws.production }
  source                          = "../../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name         = "${var.environment}-hm-tracker-kafka"
  kafka_version                   = "3.6.0"
  kafka_broker_instance_type      = "kafka.m7g.large"
  kafka_broker_number             = local.tracker_kafka_broker_number
  kafka_broker_log_s3_bucket_name = module.hm_amazon_s3_bucket_development_hm_tracker_kafka.name
  amazon_vpc_security_group_id    = "sg-xxxxxxxxxxxxxxxxx"
  amazon_vpc_subnet_ids           = local.tracker_amazon_vpc_private_subnet_ids
  aws_kms_key_arn                 = data.aws_kms_alias.aws_kms_kafka_key.arn
  environment                     = var.environment
  team                            = var.team
}
# Tracker Kafka - Kafka sink plugin
locals {
  tracker_kafka_sink_plugin_name      = "${var.environment}-hm-tracker-sink-plugin"
  tracker_kafka_sink_plugin_file_name = "${local.tracker_kafka_sink_plugin_name}.zip"
}
module "local_tracker_sink_plugin" {
  source                                         = "../../../../modules/aws/hm_local_tracker_sink_plugin"
  kafka_plugin_name                              = local.tracker_kafka_sink_plugin_name
  snowflake_kafka_connector_version              = "2.2.1"
  bc_fips_version                                = "1.0.2.4"
  bcpkix_fips_version                            = "1.0.7"
  confluent_kafka_connect_avro_converter_version = "7.6.1"
  local_dir_path                                 = "files/amazon-msk/${var.environment}-hm-tracker-kafka/plugins"
  local_file_name                                = local.tracker_kafka_sink_plugin_file_name
}
module "hm_amazon_s3_object_tracker_kafka_sink_plugin" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.hm_amazon_s3_bucket_development_hm_tracker_kafka.name
  s3_key          = "plugins/${local.tracker_kafka_sink_plugin_file_name}"
  local_file_path = module.local_tracker_sink_plugin.local_file_path
}
module "hm_amazon_msk_plugin_tracker_kafka_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = local.tracker_kafka_sink_plugin_name
  s3_bucket_arn            = module.hm_amazon_s3_bucket_development_hm_tracker_kafka.arn
  amazon_msk_plugin_s3_key = module.hm_amazon_s3_object_tracker_kafka_sink_plugin.s3_key
}
# Tracker Kafka - Kafka sink connector
locals {
  production_tracker_sink_connector_name = "DevelopmentTrackerSinkConnector"
}
module "hm_amazon_msk_tracker_sink_connector_iam" {
  providers                 = { aws = aws.production }
  source                    = "../../../../modules/aws/hm_amazon_msk_connector_iam"
  amazon_msk_connector_name = local.production_tracker_sink_connector_name
  amazon_msk_arn            = module.hm_amazon_msk_cluster.arn
  msk_plugin_s3_bucket_name = module.hm_amazon_s3_bucket_development_hm_tracker_kafka.name
  msk_log_s3_bucket_name    = module.hm_amazon_s3_bucket_development_hm_tracker_kafka.name
  environment               = var.environment
  team                      = var.team
}
data "aws_secretsmanager_secret" "tracker_snowflake_secret" {
  name = "hm/snowflake/production_hm_kafka_db/product/read_write"
}
data "aws_secretsmanager_secret_version" "tracker_snowflake_secret_version" {
  secret_id = data.aws_secretsmanager_secret.tracker_snowflake_secret.id
}
module "hm_amazon_msk_tracker_sink_connector" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/aws/hm_amazon_msk_connector"
  amazon_msk_connector_name            = local.production_tracker_sink_connector_name
  kafka_connect_version                = "2.7.1"
  amazon_msk_plugin_arn                = module.hm_amazon_msk_plugin_tracker_kafka_sink_plugin.arn
  amazon_msk_plugin_revision           = module.hm_amazon_msk_plugin_tracker_kafka_sink_plugin.latest_revision
  amazon_msk_connector_iam_role_arn    = module.hm_amazon_msk_tracker_sink_connector_iam.arn
  amazon_msk_cluster_bootstrap_servers = module.hm_amazon_msk_cluster.bootstrap_servers
  confluent_schema_registry_url        = "https://production-confluent-schema-registry.hongbomiao.com"
  snowflake_user_name                  = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["user_name"]
  snowflake_private_key                = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key"]
  snowflake_private_key_passphrase     = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key_passphrase"]
  snowflake_role_name                  = "HM_DEVELOPMENT_HM_KAFKA_DB_PRODUCT_READ_WRITE_ROLE"
  msk_log_s3_bucket_name               = module.hm_amazon_s3_bucket_development_hm_tracker_kafka.name
  msk_log_s3_key                       = "amazon-msk/connectors/${local.production_tracker_sink_connector_name}"
  kafka_topic_name                     = "production.tracker.analytic-events.avro"
  snowflake_database_name              = "DEVELOPMENT_HM_KAFKA_DB"
  snowflake_schema_name                = "ENGINEERING"
  environment                          = var.environment
  team                                 = var.team
}

# Amazon EKS
locals {
  amazon_eks_cluster_name = "hm-${var.environment}-eks-cluster"
}
module "hm_amazon_eks_access_entry_iam" {
  providers                    = { aws = aws.production }
  source                       = "../../../../modules/aws/hm_amazon_eks_access_entry_iam"
  amazon_eks_access_entry_name = "${amazon_eks_cluster_name}-access-entry"
  environment                  = var.environment
  team                         = var.team
}
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest
module "hm_amazon_eks_cluster" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "20.17.2"
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
      addon_version               = "v1.18.2-eksbuild.1"
      resolve_conflicts_on_create = "OVERWRITE"
      resolve_conflicts_on_update = "OVERWRITE"
    }
    aws-ebs-csi-driver = {
      addon_version               = "v1.32.0-eksbuild.1"
      service_account_role_arn    = "arn:aws:iam::272394222652:role/AmazonEBSCSIDriverRole-hm-eks-cluster"
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
      cidr_blocks = ["172.16.0.0/12"]
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
    }
  }
  vpc_id                   = var.amazon_vpc_id
  subnet_ids               = var.amazon_vpc_private_subnet_ids
  control_plane_subnet_ids = var.amazon_vpc_private_subnet_ids
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
    Environment  = var.environment
    Team         = var.team
    ResourceName = local.amazon_eks_cluster_name
  }
}
# https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest/submodules/karpenter
module "karpenter" {
  source       = "terraform-aws-modules/eks/aws//modules/karpenter"
  cluster_name = module.hm_amazon_eks_cluster.cluster_name
  # Attach additional IAM policies to the Karpenter node IAM role
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "${local.amazon_eks_cluster_name}-karpenter"
  }
}

# Amazon EBS CSI Driver - IAM role
module "hm_amazon_ebs_csi_driver_iam_role" {
  source                               = "../../../../modules/aws/hm_amazon_ebs_csi_driver_iam_role"
  amazon_eks_cluster_name              = module.hm_amazon_eks_cluster.cluster_name
  amazon_eks_cluster_oidc_provider     = module.hm_amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.hm_amazon_eks_cluster.oidc_provider_arn
  environment                          = var.environment
  team                                 = var.team
}

# Argo CD
module "hm_kubernetes_namespace_hm_argo_cd" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-argo-cd"
  depends_on = [
    module.hm_amazon_eks_cluster
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

# Sealed Secrets
module "hm_kubernetes_namespace_hm_sealed_secrets" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-sealed-secrets"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Traefik
module "hm_kubernetes_namespace_hm_traefik" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-traefik"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# ExternalDNS
module "hm_kubernetes_namespace_hm_external_dns" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-external-dns"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}
module "hm_external_dns_iam_role" {
  source                               = "../../../../modules/aws/hm_external_dns_iam_role"
  external_dns_service_account_name    = "hm-external-dns"
  external_dns_namespace               = "${var.environment}-hm-external-dns"
  amazon_eks_cluster_oidc_provider     = module.hm_amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.hm_amazon_eks_cluster.oidc_provider_arn
  amazon_route53_hosted_zone_id        = var.amazon_route53_hosted_zone_id
  environment                          = var.environment
  team                                 = var.team
}

# cert-manager
module "hm_cert_manager_iam_role" {
  source                               = "../../../../modules/aws/hm_cert_manager_iam_role"
  cert_manager_service_account_name    = "hm-cert-manager"
  cert_manager_namespace               = "${var.environment}-hm-cert-manager"
  amazon_eks_cluster_oidc_provider     = module.hm_amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.hm_amazon_eks_cluster.oidc_provider_arn
  amazon_route53_hosted_zone_id        = var.amazon_route53_hosted_zone_id
  amazon_route53_hosted_zone_name      = var.amazon_route53_hosted_zone_name
  environment                          = var.environment
  team                                 = var.team
}
module "hm_kubernetes_namespace_hm_cert_manager" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-cert-manager"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
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
  airbyte_postgres_name = "${var.environment}-hm-airbyte-postgres"
}
data "aws_secretsmanager_secret" "hm_airbyte_postgres_secret" {
  name = "${var.environment}-hm-airbyte-postgres/admin"
}
data "aws_secretsmanager_secret_version" "hm_airbyte_postgres_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_airbyte_postgres_secret.id
}
module "hm_airbyte_postgres_security_group" {
  source                         = "../../../../modules/aws/hm_amazon_rds_security_group"
  amazon_ec2_security_group_name = "${local.airbyte_postgres_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.hm_terraform_remote_state_production_aws_network.outputs.hm_amazon_vpc_id
  environment                    = var.environment
  team                           = var.team
}
module "hm_airbyte_postgres_subnet_group" {
  source            = "../../../../modules/aws/hm_amazon_rds_subnet_group"
  subnet_group_name = "${local.airbyte_postgres_name}-subnet-group"
  subnet_ids        = var.amazon_vpc_private_subnet_ids
  environment       = var.environment
  team              = var.team
}
module "hm_airbyte_postgres_parameter_group" {
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
module "hm_airbyte_postgres_instance" {
  source                     = "../../../../modules/aws/hm_amazon_rds_instance"
  amazon_rds_name            = local.airbyte_postgres_name
  amazon_rds_engine          = "postgres"
  amazon_rds_engine_version  = "16.3"
  amazon_rds_instance_class  = "db.m7g.large"
  amazon_rds_storage_size_gb = 32
  user_name                  = jsondecode(data.aws_secretsmanager_secret_version.hm_airbyte_postgres_secret_version.secret_string)["user_name"]
  password                   = jsondecode(data.aws_secretsmanager_secret_version.hm_airbyte_postgres_secret_version.secret_string)["password"]
  parameter_group_name       = module.hm_airbyte_postgres_parameter_group.name
  subnet_group_name          = module.hm_airbyte_postgres_subnet_group.name
  vpc_security_group_ids     = [module.hm_airbyte_postgres_security_group.id]
  cloudwatch_log_types       = ["postgresql", "upgrade"]
  environment                = var.environment
  team                       = var.team
}
module "hm_kubernetes_namespace_hm_airbyte" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-airbyte"
  labels = {
    "goldilocks.fairwinds.com/enabled" = true
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# MLFlow
# MLFlow - S3 bucket
module "hm_amazon_s3_bucket_hm_mlflow" {
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${var.environment}-hm-mlflow"
  environment    = var.environment
  team           = var.team
}
# MLFlow - IAM role
module "hm_mlflow_iam_role" {
  source                               = "../../../../modules/aws/hm_mlflow_iam_role"
  mlflow_service_account_name          = "hm-mlflow"
  mlflow_namespace                     = "${var.environment}-hm-mlflow"
  amazon_eks_cluster_oidc_provider     = module.hm_amazon_eks_cluster.oidc_provider
  amazon_eks_cluster_oidc_provider_arn = module.hm_amazon_eks_cluster.oidc_provider_arn
  s3_bucket_name                       = module.hm_amazon_s3_bucket_hm_mlflow.name
  environment                          = var.environment
  team                                 = var.team
}
# MLFlow - Postgres
locals {
  mlflow_postgres_name = "${var.environment}-hm-mlflow-postgres"
}
data "aws_secretsmanager_secret" "hm_mlflow_postgres_secret" {
  name = "${var.environment}-hm-mlflow-postgres/admin"
}
data "aws_secretsmanager_secret_version" "hm_mlflow_postgres_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_mlflow_postgres_secret.id
}
module "hm_mlflow_postgres_security_group" {
  source                         = "../../../../modules/aws/hm_amazon_rds_security_group"
  amazon_ec2_security_group_name = "${local.mlflow_postgres_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.hm_terraform_remote_state_production_aws_network.outputs.hm_amazon_vpc_id
  environment                    = var.environment
  team                           = var.team
}
module "hm_mlflow_postgres_subnet_group" {
  source            = "../../../../modules/aws/hm_amazon_rds_subnet_group"
  subnet_group_name = "${local.mlflow_postgres_name}-subnet-group"
  subnet_ids        = var.amazon_vpc_private_subnet_ids
  environment       = var.environment
  team              = var.team
}
module "hm_mlflow_postgres_parameter_group" {
  source               = "../../../../modules/aws/hm_amazon_rds_parameter_group"
  family               = "postgres16"
  parameter_group_name = "${local.mlflow_postgres_name}-parameter-group"
  environment          = var.environment
  team                 = var.team
}
module "hm_mlflow_postgres_instance" {
  source                     = "../../../../modules/aws/hm_amazon_rds_instance"
  amazon_rds_name            = local.mlflow_postgres_name
  amazon_rds_engine          = "postgres"
  amazon_rds_engine_version  = "16.3"
  amazon_rds_instance_class  = "db.m7g.large"
  amazon_rds_storage_size_gb = 32
  user_name                  = jsondecode(data.aws_secretsmanager_secret_version.hm_mlflow_postgres_secret_version.secret_string)["user_name"]
  password                   = jsondecode(data.aws_secretsmanager_secret_version.hm_mlflow_postgres_secret_version.secret_string)["password"]
  parameter_group_name       = module.hm_mlflow_postgres_parameter_group.name
  subnet_group_name          = module.hm_mlflow_postgres_subnet_group.name
  vpc_security_group_ids     = [module.hm_mlflow_postgres_security_group.id]
  cloudwatch_log_types       = ["postgresql", "upgrade"]
  environment                = var.environment
  team                       = var.team
}
module "hm_kubernetes_namespace_hm_mlflow" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-mlflow"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Vertical Pod Autoscaler
module "hm_kubernetes_namespace_hm_vertical_pod_autoscaler" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-vertical-pod-autoscaler"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Goldilocks
module "hm_kubernetes_namespace_hm_goldilocks" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-goldilocks"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Metrics Server
module "hm_kubernetes_namespace_hm_metrics_server" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-metrics-server"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Prometheus
module "hm_kubernetes_namespace_hm_prometheus" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-prometheus"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# Netdata
module "hm_kubernetes_namespace_hm_netdata" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-netdata"
  labels = {
    "goldilocks.fairwinds.com/enabled" = true
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}

# OpenCost
module "hm_kubernetes_namespace_hm_opencost" {
  source               = "../../../../modules/kubernetes/hm_kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-opencost"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_amazon_eks_cluster
  ]
}
