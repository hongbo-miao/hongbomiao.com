terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/kms_key
resource "aws_kms_key" "hm_aws_kms_key" {
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.aws_kms_key_name
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/kms_alias
resource "aws_kms_alias" "hm_aws_kms_key_alias" {
  name          = "alias/${var.aws_kms_key_name}"
  target_key_id = aws_kms_key.hm_aws_kms_key.key_id
}
