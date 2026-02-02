terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/grant_account_role
resource "snowflake_grant_account_role" "grant_role_to_user" {
  role_name = var.snowflake_role_name
  user_name = var.snowflake_user_name
}
