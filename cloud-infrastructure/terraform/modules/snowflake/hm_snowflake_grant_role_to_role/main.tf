terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_account_role
resource "snowflake_grant_account_role" "hm_snowflake_grant_role_to_role" {
  role_name        = var.snowflake_role_name
  parent_role_name = var.snowflake_parent_role_name
}
