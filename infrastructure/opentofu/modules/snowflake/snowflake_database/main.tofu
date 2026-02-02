terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/database
resource "snowflake_database" "main" {
  name                        = var.snowflake_database_name
  data_retention_time_in_days = var.data_retention_days
  lifecycle {
    prevent_destroy = true
  }
}
