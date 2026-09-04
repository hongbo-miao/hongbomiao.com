terraform {
  backend "s3" {
    bucket  = "production-hm-opentofu"
    key     = "production/zitadel/terraform.tfstate"
    region  = "us-central1"
    profile = "nebius"
    endpoints = {
      "s3" = "https://storage.us-central1.nebius.cloud"
    }
    # Skip AWS-specific validations as Nebius provides different settings
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
    skip_region_validation      = true
  }
  required_providers {
    # https://search.opentofu.org/provider/zitadel/zitadel/latest
    zitadel = {
      source  = "zitadel/zitadel"
      version = "3.3.0"
    }
  }
  required_version = ">= 1.11"
}
