terraform {
  backend "s3" {
    bucket  = "production-hm-opentofu-bucket"
    key     = "production/grafana/terraform.tfstate"
    region  = "us-central1"
    profile = "nebius"
    endpoints = {
      "s3" = "https://storage.us-central1.nebius.cloud"
    }
    # Skip AWS-specific validations as Nebius provides different settings
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
  required_providers {
    # https://search.opentofu.org/provider/grafana/grafana/latest
    grafana = {
      source  = "grafana/grafana"
      version = "4.25.0"
    }
    # https://search.opentofu.org/provider/hashicorp/http/latest
    http = {
      source  = "hashicorp/http"
      version = "3.5.0"
    }
  }
  required_version = ">= 1.11"
}
