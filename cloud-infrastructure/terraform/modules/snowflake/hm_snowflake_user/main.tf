terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/user
resource "snowflake_user" "main" {
  name              = var.snowflake_user_name
  default_role      = var.default_role
  default_warehouse = var.default_warehouse
  rsa_public_key    = var.rsa_public_key
}
