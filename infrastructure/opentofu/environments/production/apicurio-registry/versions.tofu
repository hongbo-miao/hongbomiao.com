terraform {
  backend "s3" {
    bucket  = "production-hm-opentofu"
    key     = "production/apicurio-registry/terraform.tfstate"
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
    // https://registry.terraform.io/providers/tae2089/apicurio-registry/latest
    apicurio = {
      source  = "registry.terraform.io/tae2089/apicurio-registry"
      version = "1.0.0"
    }
  }
  required_version = ">= 1.11"
}
