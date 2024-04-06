terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/role
resource "snowflake_role" "hm_snowflake_role" {
  name = var.snowflake_role_name
}
