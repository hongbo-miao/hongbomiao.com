terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/schema
resource "snowflake_schema" "main" {
  database                    = var.snowflake_database_name
  name                        = var.snowflake_schema_name
  data_retention_time_in_days = var.data_retention_days
  with_managed_access         = var.with_managed_access
  is_transient                = false
  lifecycle {
    prevent_destroy = true
  }
}
