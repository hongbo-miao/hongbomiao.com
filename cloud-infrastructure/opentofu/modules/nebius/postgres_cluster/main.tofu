terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/msp_postgresql_v1alpha1_cluster
resource "nebius_msp_postgresql_v1alpha1_cluster" "main" {
  parent_id  = var.project_id
  network_id = var.network_id
  name       = var.postgres_cluster_name
  bootstrap = {
    db_name       = var.bootstrap_database_name
    user_name     = var.bootstrap_user_name
    user_password = var.bootstrap_user_password
  }
  config = {
    version       = var.postgres_version
    public_access = false
    template = {
      hosts = {
        count = var.host_count
      }
      resources = {
        platform = var.platform
        preset   = var.preset
      }
      disk = {
        size_gibibytes = var.disk_size_gib
        type           = "network-ssd"
      }
    }
    pooler_config = {
      pooling_mode  = "TRANSACTION"
      max_pool_size = 100
    }
  }
  backup = {
    backup_window_start = var.backup_window_start_time_utc
    retention_policy    = var.backup_retention_policy
  }
  labels = var.labels
  lifecycle {
    prevent_destroy = true
  }
}
