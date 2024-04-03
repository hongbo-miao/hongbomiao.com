terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_object
resource "aws_s3_object" "hm_amazon_s3_object" {
  bucket = var.amazon_s3_bucket
  key    = var.amazon_s3_key
  source = var.local_file_path
  etag   = filemd5(var.local_file_path)
}
