data "nebius_iam_v1_group" "editors_group" {
  name      = "editors"
  parent_id = var.tenant_id
}

module "hm_kubernetes_cluster" {
  source                  = "../../../../modules/nebius/kubernetes_cluster"
  project_id              = var.project_id
  kubernetes_cluster_name = "hm-kubernetes-cluster"
  vpc_subnet_id           = var.vpc_subnet_id
  labels                  = var.common_labels
}
module "hm_kubernetes_cpu_node_group_lion" {
  source         = "../../../../modules/nebius/kubernetes_cpu_node_group"
  parent_id      = module.hm_kubernetes_cluster.id
  name           = "node-group-lion"
  min_node_count = 3
  max_node_count = 100
  # https://docs.nebius.com/compute/virtual-machines/types
  platform = "cpu-d3"
  preset   = "4vcpu-16gb"
  # https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_node_group#inner-value-description
  os     = "ubuntu24.04"
  labels = var.common_labels
}
module "hm_kubernetes_gpu_node_group_dragon" {
  source         = "../../../../modules/nebius/kubernetes_gpu_node_group"
  parent_id      = module.hm_kubernetes_cluster.id
  name           = "node-group-dragon"
  min_node_count = 1
  max_node_count = 10
  # https://docs.nebius.com/compute/virtual-machines/types
  platform = "gpu-h200-sxm"
  preset   = "1gpu-16vcpu-200gb"
  # https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_node_group#inner-value-description
  os             = "ubuntu24.04"
  drivers_preset = "cuda12.8"
  labels         = var.common_labels
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

# Harbor
module "editors_group_member_hm_harbor_service_account" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = data.nebius_iam_v1_group.editors_group.id
  member_id = module.hm_harbor_service_account.id
  labels    = var.common_labels
}
module "hm_harbor_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "harbor-service-account"
  labels               = var.common_labels
}
module "hm_harbor_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-harbor"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
module "harbor_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  postgres_cluster_name        = "${var.environment}-harbor-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.harbor_postgres_bootstrap_user_password
  public_access                = true
  disk_size_gib                = 16
  host_count                   = 1
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "10:00:00"
  backup_retention_policy      = "14d"
}

# MLflow
module "editors_group_member_hm_mlflow_service_account" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = data.nebius_iam_v1_group.editors_group.id
  member_id = module.hm_mlflow_service_account.id
  labels    = var.common_labels
}
module "hm_mlflow_service_account" {
  source               = "../../../../modules/nebius/service_account"
  project_id           = var.project_id
  service_account_name = "mlflow-service-account"
  labels               = var.common_labels
}
module "hm_mlflow_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-mlflow"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
module "mlflow_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  postgres_cluster_name        = "${var.environment}-mlflow-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.mlflow_postgres_bootstrap_user_password
  public_access                = true
  disk_size_gib                = 16
  host_count                   = 1
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "10:00:00"
  backup_retention_policy      = "14d"
}
