terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_ownership
resource "snowflake_grant_ownership" "hm_snowflake_transfer_schema_ownership_to_role" {
  account_role_name   = var.snowflake_role_name
  outbound_privileges = "REVOKE"
  on {
    object_type = "SCHEMA"
    object_name = "\"${var.snowflake_database_name}\".\"${var.snowflake_schema_name}\""
  }
}
