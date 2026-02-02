terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/kms_key
resource "aws_kms_key" "main" {
  tags = merge(var.common_tags, {
    "hm_resource_name" = var.aws_kms_key_name
  })
}
# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/kms_alias
resource "aws_kms_alias" "main" {
  name          = "alias/${var.aws_kms_key_name}"
  target_key_id = aws_kms_key.main.key_id
}
