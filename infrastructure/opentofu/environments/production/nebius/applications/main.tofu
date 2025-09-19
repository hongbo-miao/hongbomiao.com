data "nebius_iam_v1_group" "editors_group" {
  name      = "editors"
  parent_id = var.tenant_id
}

# MLflow
module "editors_group_member_hm_mlflow_service_account" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = data.nebius_iam_v1_group.editors_group.id
  member_id = module.hm_mlflow_service_account.id
  labels    = var.common_labels
}
module "hm_mlflow_service_account" {
  source               = "../../../../modules/nebius/iam_service_account"
  project_id           = var.project_id
  service_account_name = "mlflow-service-account"
  labels               = var.common_labels
}
module "hm_mlflow_data_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-mlflow-data"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
module "hm_mlflow_cluster" {
  source              = "../../../../modules/nebius/mlflow_cluster"
  project_id          = var.project_id
  network_id          = var.vpc_network_id
  service_account_id  = module.hm_mlflow_service_account.id
  mlflow_cluster_name = "${var.environment}-hm-mlflow-cluster"
  mlflow_cluster_size = "medium"
  admin_username      = "mlflow_admin_dolphin"
  admin_password      = var.hm_mlflow_admin_password
  public_access       = false
  storage_bucket_name = module.hm_mlflow_data_object_storage.name
  labels              = var.common_labels
}
