terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/key_pair
resource "aws_key_pair" "main" {
  key_name   = var.key_name
  public_key = var.public_key
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.key_name
  })
}
