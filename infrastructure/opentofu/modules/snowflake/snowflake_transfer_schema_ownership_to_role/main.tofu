terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/grant_ownership
resource "snowflake_grant_ownership" "transfer_schema_ownership_to_role" {
  account_role_name   = var.snowflake_role_name
  outbound_privileges = "REVOKE"
  on {
    object_type = "SCHEMA"
    object_name = "\"${var.snowflake_database_name}\".\"${var.snowflake_schema_name}\""
  }
}
