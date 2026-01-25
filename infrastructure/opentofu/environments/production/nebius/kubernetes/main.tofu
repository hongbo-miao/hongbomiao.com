# Kubernetes
module "hm_kubernetes_cluster" {
  source                  = "../../../../modules/nebius/kubernetes_cluster"
  project_id              = var.project_id
  kubernetes_cluster_name = "hm-kubernetes-cluster"
  vpc_subnet_id           = var.vpc_subnet_id
  labels                  = var.common_labels
}

# Kubernetes - CPU node groups
# https://docs.nebius.com/compute/virtual-machines/types
# https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_node_group#inner-value-description
locals {
  cloud_init_user_data = file("${path.module}/files/scripts/configure_sysctl.yaml")
  cpu_node_groups = {
    lion = {
      min_node_count = 3
      max_node_count = 100
      preset         = "4vcpu-16gb"
    },
    badger = {
      min_node_count = 3
      max_node_count = 50
      preset         = "8vcpu-32gb"
    },
    hyena = {
      min_node_count = 0
      max_node_count = 20
      preset         = "16vcpu-64gb"
    },
    dingo = {
      min_node_count = 0
      max_node_count = 10
      preset         = "32vcpu-128gb"
    }
  }
}
module "hm_kubernetes_cpu_node_group" {
  for_each             = local.cpu_node_groups
  source               = "../../../../modules/nebius/kubernetes_cpu_node_group"
  parent_id            = module.hm_kubernetes_cluster.id
  name                 = "node-group-${each.key}"
  min_node_count       = each.value.min_node_count
  max_node_count       = each.value.max_node_count
  platform             = "cpu-d3"
  preset               = each.value.preset
  os                   = "ubuntu24.04"
  cloud_init_user_data = local.cloud_init_user_data
  labels               = var.common_labels
}

# Kubernetes - GPU node groups
locals {
  gpu_node_groups = {
    dragon = {
      min_node_count = 0
      max_node_count = 10
      platform       = "gpu-h200-sxm"
      preset         = "1gpu-16vcpu-200gb"
      drivers_preset = "cuda12.8"
    }
  }
}
module "hm_kubernetes_gpu_node_group" {
  for_each             = local.gpu_node_groups
  source               = "../../../../modules/nebius/kubernetes_gpu_node_group"
  parent_id            = module.hm_kubernetes_cluster.id
  name                 = "node-group-${each.key}"
  min_node_count       = each.value.min_node_count
  max_node_count       = each.value.max_node_count
  platform             = each.value.platform
  preset               = each.value.preset
  os                   = "ubuntu24.04"
  drivers_preset       = each.value.drivers_preset
  cloud_init_user_data = local.cloud_init_user_data
  labels               = var.common_labels
}

# Argo CD
# Argo CD - Kubernetes namespace
module "kubernetes_namespace_hm_argo_cd" {
  source               = "../../../../modules/kubernetes/kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-argo-cd"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_kubernetes_cluster
  ]
}
# Argo CD - Helm chart
module "argo_cd_helm_chart" {
  source              = "../../../../modules/kubernetes/helm_chart"
  repository          = "https://argoproj.github.io/argo-helm"
  chart_name          = "argo-cd"
  chart_version       = "8.3.1" # https://artifacthub.io/packages/helm/argo/argo-cd
  release_name        = "argo-cd"
  namespace           = module.kubernetes_namespace_hm_argo_cd.namespace
  my_values_yaml_path = "files/argo-cd/helm/my-values.yaml"
  depends_on = [
    module.kubernetes_namespace_hm_argo_cd
  ]
}

# Grafana
# Grafana - service account
module "hm_grafana_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "grafana-service-account"
  labels               = var.common_labels
}
module "hm_grafana_iam_group" {
  source     = "../../../../modules/nebius/iam_group"
  parent_id  = var.tenant_id
  group_name = "${var.environment}-grafana-group"
  labels     = var.common_labels
}
module "hm_grafana_iam_group_membership" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = module.hm_grafana_iam_group.id
  member_id = module.hm_grafana_service_account.id
  labels    = var.common_labels
}
module "hm_grafana_access_permit" {
  source      = "../../../../modules/nebius/access_permit"
  parent_id   = module.hm_grafana_iam_group.id
  resource_id = module.hm_grafana_object_storage.id
  role        = "storage.object-editor"
  labels      = var.common_labels
}
# Grafana - object storage
module "hm_grafana_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-grafana"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
# Grafana - Postgres
module "grafana_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  pooling_mode                 = "SESSION"
  postgres_cluster_name        = "${var.environment}-grafana-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.grafana_postgres_bootstrap_user_password
  public_access                = false
  disk_size_gib                = 16
  host_count                   = 1
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "9:00:00"
  backup_retention_policy      = "14d"
}

# Harbor
# Harbor - service account
module "hm_harbor_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "harbor-service-account"
  labels               = var.common_labels
}
module "hm_harbor_iam_group" {
  source     = "../../../../modules/nebius/iam_group"
  parent_id  = var.tenant_id
  group_name = "${var.environment}-harbor-group"
  labels     = var.common_labels
}
module "hm_harbor_iam_group_membership" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = module.hm_harbor_iam_group.id
  member_id = module.hm_harbor_service_account.id
  labels    = var.common_labels
}
module "hm_harbor_access_permit" {
  source      = "../../../../modules/nebius/access_permit"
  parent_id   = module.hm_harbor_iam_group.id
  resource_id = module.hm_harbor_object_storage.id
  role        = "storage.object-editor"
  labels      = var.common_labels
}
# Harbor - object storage
module "hm_harbor_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-harbor"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
# Harbor - Postgres
module "harbor_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  pooling_mode                 = "SESSION"
  postgres_cluster_name        = "${var.environment}-harbor-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.harbor_postgres_bootstrap_user_password
  public_access                = false
  disk_size_gib                = 16
  host_count                   = 1
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "9:00:00"
  backup_retention_policy      = "14d"
}


# Loki
# Loki - service account
module "hm_loki_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "${var.environment}-loki-service-account"
  labels               = var.common_labels
}
module "hm_loki_iam_group" {
  source     = "../../../../modules/nebius/iam_group"
  parent_id  = var.tenant_id
  group_name = "${var.environment}-loki-group"
  labels     = var.common_labels
}
module "hm_loki_iam_group_membership" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = module.hm_loki_iam_group.id
  member_id = module.hm_loki_service_account.id
  labels    = var.common_labels
}
locals {
  loki_buckets = toset(["admin", "chunk", "ruler"])
}
module "hm_loki_access_permit" {
  for_each    = local.loki_buckets
  source      = "../../../../modules/nebius/access_permit"
  parent_id   = module.hm_loki_iam_group.id
  resource_id = module.hm_loki_object_storage[each.key].id
  role        = "storage.object-editor"
  labels      = var.common_labels
}
# Loki - object storage
module "hm_loki_object_storage" {
  for_each              = local.loki_buckets
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-loki-${each.key}"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}

# MLflow
# MLflow - service account
module "hm_mlflow_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "mlflow-service-account"
  labels               = var.common_labels
}
module "hm_mlflow_iam_group" {
  source     = "../../../../modules/nebius/iam_group"
  parent_id  = var.tenant_id
  group_name = "${var.environment}-mlflow-group"
  labels     = var.common_labels
}
module "hm_mlflow_iam_group_membership" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = module.hm_mlflow_iam_group.id
  member_id = module.hm_mlflow_service_account.id
  labels    = var.common_labels
}
module "hm_mlflow_access_permit" {
  source      = "../../../../modules/nebius/access_permit"
  parent_id   = module.hm_mlflow_iam_group.id
  resource_id = module.hm_mlflow_object_storage.id
  role        = "storage.object-editor"
  labels      = var.common_labels
}
# MLflow - object storage
module "hm_mlflow_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-mlflow"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
# MLflow - Postgres
module "mlflow_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  pooling_mode                 = "SESSION"
  postgres_cluster_name        = "${var.environment}-mlflow-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.mlflow_postgres_bootstrap_user_password
  public_access                = false
  disk_size_gib                = 16
  host_count                   = 1
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "9:00:00"
  backup_retention_policy      = "14d"
}

# SkyPilot
# SkyPilot - Postgres
module "skypilot_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  pooling_mode                 = "SESSION"
  postgres_cluster_name        = "${var.environment}-skypilot-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.skypilot_postgres_bootstrap_user_password
  public_access                = false
  disk_size_gib                = 16
  host_count                   = 1
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "9:00:00"
  backup_retention_policy      = "14d"
}

# Spark History Server
# Spark History Server - service account
module "hm_spark_history_server_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "${var.environment}-spark-history-server-service-account"
  labels               = var.common_labels
}
module "hm_spark_history_server_iam_group" {
  source     = "../../../../modules/nebius/iam_group"
  parent_id  = var.tenant_id
  group_name = "${var.environment}-spark-history-server-group"
  labels     = var.common_labels
}
module "hm_spark_history_server_iam_group_membership" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = module.hm_spark_history_server_iam_group.id
  member_id = module.hm_spark_history_server_service_account.id
  labels    = var.common_labels
}
module "hm_spark_history_server_access_permit" {
  source      = "../../../../modules/nebius/access_permit"
  parent_id   = module.hm_spark_history_server_iam_group.id
  resource_id = module.hm_spark_history_server_object_storage.id
  role        = "storage.object-editor"
  labels      = var.common_labels
}
# Spark History Server - object storage
module "hm_spark_history_server_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-spark-history-server"
  default_storage_class = "STANDARD"
  versioning_policy     = "DISABLED"
  labels                = var.common_labels
}
