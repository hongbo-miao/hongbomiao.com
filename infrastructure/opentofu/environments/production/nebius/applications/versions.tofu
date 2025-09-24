terraform {
  backend "s3" {
    bucket  = "production-hm-opentofu-bucket"
    key     = "production/nebius/applications/terraform.tfstate"
    region  = "us-central1"
    profile = "nebius"
    endpoints = {
      "s3" = "https://storage.us-central1.nebius.cloud"
    }
    # Skip AWS-specific validations, since Nebius uses different settings.
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
    skip_region_validation      = true
  }
  required_providers {
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = "0.5.111"
    }
  }
  required_version = ">= 1.10"
}
