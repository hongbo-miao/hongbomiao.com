terraform {
  required_providers {
    snowflake = {
      source = "registry.opentofu.org/snowflakedb/snowflake"
    }
  }
}

# https://search.opentofu.org/provider/snowflakedb/snowflake/latest/docs/resources/account_parameter
resource "snowflake_account_parameter" "main" {
  key   = var.key
  value = var.value
}
