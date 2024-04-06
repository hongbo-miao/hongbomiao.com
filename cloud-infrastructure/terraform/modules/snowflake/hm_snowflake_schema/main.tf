terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/schema
resource "snowflake_schema" "hm_snowflake_schema" {
  database            = var.snowflake_database_name
  name                = var.snowflake_schema_name
  is_managed          = false
  is_transient        = false
  data_retention_days = 7
}
