terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/kms_key
resource "aws_kms_key" "main" {
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.aws_kms_key_name
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/kms_alias
resource "aws_kms_alias" "main" {
  name          = "alias/${var.aws_kms_key_name}"
  target_key_id = aws_kms_key.main.key_id
}
