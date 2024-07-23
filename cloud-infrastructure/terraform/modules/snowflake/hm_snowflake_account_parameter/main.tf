terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/account_parameter
resource "snowflake_account_parameter" "hm_snowflake_account_parameter" {
  key   = var.key
  value = var.value
}
