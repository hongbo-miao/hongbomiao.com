terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/instance
resource "aws_instance" "main" {
  ami                    = var.instance_ami
  instance_type          = var.instance_type
  iam_instance_profile   = var.instance_profile
  key_name               = var.key_name
  subnet_id              = var.amazon_vpc_subnet_id
  vpc_security_group_ids = [var.amazon_ec2_security_group_id]
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.instance_name,
    Name               = var.instance_name
  })
}
