terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_schema_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  privileges        = var.privileges
  on_schema {
    schema_name = "\"${var.snowflake_database_name}\".\"${var.snowflake_schema_name}\""
  }
}
