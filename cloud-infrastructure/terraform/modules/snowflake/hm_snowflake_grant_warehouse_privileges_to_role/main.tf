terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "hm_snowflake_grant_warehouse_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  privileges        = var.privileges
  on_account_object {
    object_type = "WAREHOUSE"
    object_name = var.snowflake_warehouse_name
  }
}
