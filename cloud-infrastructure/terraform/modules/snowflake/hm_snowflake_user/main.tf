terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/user
resource "snowflake_user" "hm_snowflake_user" {
  name           = var.snowflake_user_name
  default_role   = var.default_role
  rsa_public_key = var.rsa_public_key_without_header_and_trailer
}
