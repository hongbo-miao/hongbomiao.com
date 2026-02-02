terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/s3_object
resource "aws_s3_object" "main" {
  bucket = var.s3_bucket_name
  key    = var.s3_key
  source = var.local_file_path
  etag   = filemd5(var.local_file_path)
}
