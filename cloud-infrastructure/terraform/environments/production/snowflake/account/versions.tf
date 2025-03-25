terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/snowflake/account/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest
    snowflake = {
      source  = "Snowflake-Labs/snowflake"
      version = "1.0.5"
    }
  }
  required_version = ">= 1.8"
}
