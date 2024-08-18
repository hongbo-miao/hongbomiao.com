terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_existing_schema_all_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  all_privileges    = true
  on_schema {
    all_schemas_in_database = var.snowflake_database_name
  }
}
# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_future_schema_all_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  all_privileges    = true
  on_schema {
    future_schemas_in_database = var.snowflake_database_name
  }
}
