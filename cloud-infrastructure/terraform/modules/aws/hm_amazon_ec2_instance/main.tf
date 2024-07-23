terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/instance
resource "aws_instance" "hm_amazon_ec2_instance" {
  ami                    = var.instance_ami
  instance_type          = var.instance_type
  iam_instance_profile   = var.instance_profile
  key_name               = var.key_name
  subnet_id              = var.amazon_vpc_subnet_id
  vpc_security_group_ids = [var.amazon_ec2_security_group_id]
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.instance_name
    Name         = var.instance_name
  }
}
