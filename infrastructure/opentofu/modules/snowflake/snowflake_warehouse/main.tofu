terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/warehouse
resource "snowflake_warehouse" "main" {
  name                                = var.snowflake_warehouse_name
  warehouse_size                      = var.snowflake_warehouse_size
  auto_suspend                        = var.auto_suspend_min * 60 # seconds
  auto_resume                         = "true"
  enable_query_acceleration           = "false"
  warehouse_type                      = "STANDARD"
  max_concurrency_level               = 8
  min_cluster_count                   = 1
  max_cluster_count                   = 1
  statement_timeout_in_seconds        = 108000 # 30 min
  statement_queued_timeout_in_seconds = 60     # 1 min
}
