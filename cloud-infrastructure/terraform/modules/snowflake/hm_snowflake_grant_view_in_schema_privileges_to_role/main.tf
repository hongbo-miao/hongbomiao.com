terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_existing_view_in_schema_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  privileges        = var.privileges
  on_schema_object {
    all {
      object_type_plural = "VIEWS"
      in_schema          = "\"${var.snowflake_database_name}\".\"${var.snowflake_schema_name}\""
    }
  }
}
# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/grant_privileges_to_account_role
resource "snowflake_grant_privileges_to_account_role" "grant_future_view_in_schema_privileges_to_role" {
  account_role_name = var.snowflake_role_name
  privileges        = var.privileges
  on_schema_object {
    future {
      object_type_plural = "VIEWS"
      in_schema          = "\"${var.snowflake_database_name}\".\"${var.snowflake_schema_name}\""
    }
  }
}
