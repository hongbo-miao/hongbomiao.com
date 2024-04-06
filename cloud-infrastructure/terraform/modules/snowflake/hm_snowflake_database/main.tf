terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/database
resource "snowflake_database" "hm_snowflake_database" {
  name = var.snowflake_database_name
}
