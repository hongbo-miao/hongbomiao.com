terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_existing_view_all_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  all_privileges    = true
  on_schema_object {
    all {
      object_type_plural = "VIEWS"
      in_database        = var.snowflake_database_name
    }
  }
}
# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_future_view_all_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  all_privileges    = true
  on_schema_object {
    future {
      object_type_plural = "VIEWS"
      in_database        = var.snowflake_database_name
    }
  }
}
