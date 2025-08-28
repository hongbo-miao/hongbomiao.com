terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group
resource "aws_security_group" "batch_security_group" {
  name   = var.amazon_ec2_security_group_name
  vpc_id = var.amazon_vpc_id
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_ec2_security_group_name
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_egress_rule
resource "aws_vpc_security_group_egress_rule" "egress_allow" {
  security_group_id = aws_security_group.batch_security_group.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1" # all ports
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_ec2_security_group_name
  })
}
