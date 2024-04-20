terraform {
  required_providers {
    snowflake = {
      source = "Snowflake-Labs/snowflake"
    }
  }
}

# https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest/docs/resources/warehouse
resource "snowflake_warehouse" "hm_snowflake_warehouse" {
  name           = var.snowflake_warehouse_name
  warehouse_size = var.snowflake_warehouse_size
  auto_suspend   = var.auto_suspend_min * 60 # seconds
}
