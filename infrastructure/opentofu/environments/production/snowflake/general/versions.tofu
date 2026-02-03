terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/snowflake/general/terraform.tfstate"
  }
  required_providers {
    # https://search.opentofu.org/provider/snowflakedb/snowflake/latest
    snowflake = {
      source  = "snowflakedb/snowflake"
      version = "2.12.0"
    }
  }
  required_version = ">= 1.11"
}
