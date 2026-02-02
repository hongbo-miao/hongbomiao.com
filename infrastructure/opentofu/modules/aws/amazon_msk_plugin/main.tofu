terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/mskconnect_custom_plugin
resource "aws_mskconnect_custom_plugin" "main" {
  name         = var.amazon_msk_plugin_name
  content_type = "ZIP"
  location {
    s3 {
      bucket_arn = var.s3_bucket_arn
      file_key   = var.amazon_msk_plugin_s3_key
    }
  }
}
