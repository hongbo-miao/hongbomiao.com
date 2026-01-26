module "hm_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-bucket"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}

module "hm_postgres_cluster" {
  source                       = "../../../../modules/nebius/postgres_cluster"
  project_id                   = var.project_id
  network_id                   = var.vpc_network_id
  postgres_version             = "16"
  pooling_mode                 = "TRANSACTION"
  postgres_cluster_name        = "${var.environment}-hm-postgres-cluster"
  bootstrap_database_name      = "bootstrap_db"
  bootstrap_user_name          = "bootstrap_user"
  bootstrap_user_password      = var.hm_postgres_bootstrap_user_password
  public_access                = false
  disk_size_gib                = 32
  host_count                   = 3
  platform                     = "cpu-d3"
  preset                       = "4vcpu-16gb"
  labels                       = var.common_labels
  backup_window_start_time_utc = "9:00:00"
  backup_retention_policy      = "14d"
}

module "hm_iceberg_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-iceberg"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
