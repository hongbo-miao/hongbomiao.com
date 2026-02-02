terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_warehouse_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  privileges        = var.privileges
  on_account_object {
    object_type = "WAREHOUSE"
    object_name = var.snowflake_warehouse_name
  }
}
