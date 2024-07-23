terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/key_pair
resource "aws_key_pair" "hm_amazon_ec2_key_pair" {
  key_name   = var.key_name
  public_key = var.public_key
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.key_name
  }
}
