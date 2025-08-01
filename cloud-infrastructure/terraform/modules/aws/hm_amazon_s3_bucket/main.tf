terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket
resource "aws_s3_bucket" "main" {
  bucket = var.s3_bucket_name
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.s3_bucket_name
  })
  lifecycle {
    prevent_destroy = true
  }
}
