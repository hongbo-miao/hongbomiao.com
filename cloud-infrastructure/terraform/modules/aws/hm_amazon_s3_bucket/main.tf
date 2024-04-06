terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket
resource "aws_s3_bucket" "hm_amazon_s3_bucket" {
  bucket = var.s3_bucket_name
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.s3_bucket_name
  }
}
