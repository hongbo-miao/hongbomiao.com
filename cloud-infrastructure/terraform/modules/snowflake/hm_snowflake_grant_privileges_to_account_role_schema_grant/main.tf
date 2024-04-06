terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_privileges_to_account_role
# https://quickstarts.snowflake.com/guide/terraforming_snowflake/index.html#8
resource "snowflake_grant_privileges_to_account_role" "hm_snowflake_grant_privileges_to_account_role_schema_grant" {
  account_role_name = var.account_role_name
  privileges        = var.privileges
  on_schema {
    schema_name = "\"${var.snowflake_database_name}\".\"${var.snowflake_schema_name}\""
  }
}
