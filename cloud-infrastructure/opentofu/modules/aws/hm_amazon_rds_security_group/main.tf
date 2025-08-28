terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group
resource "aws_security_group" "rds_security_group" {
  name   = var.amazon_ec2_security_group_name
  vpc_id = var.amazon_vpc_id
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_ec2_security_group_name
  })
}
# Ingress - On-Site
locals {
  ingress_rule_on_site = "on_site"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_ingress_rule
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_on_site" {
  security_group_id = aws_security_group.rds_security_group.id
  description       = local.ingress_rule_on_site
  cidr_ipv4         = "10.10.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 5432
  to_port           = 5432
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_on_site
  })
}
locals {
  ingress_rule_vpn = "vpn"
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpn" {
  security_group_id = aws_security_group.rds_security_group.id
  description       = "VPN"
  cidr_ipv4         = "10.100.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 5432
  to_port           = 5432
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpn
  })
}
locals {
  ingress_rule_vpc = "vpc"
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpc" {
  security_group_id = aws_security_group.rds_security_group.id
  description       = local.ingress_rule_vpc
  cidr_ipv4         = var.amazon_vpc_cidr_ipv4
  ip_protocol       = "tcp"
  from_port         = 5432
  to_port           = 5432
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpc
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_egress_rule
resource "aws_vpc_security_group_egress_rule" "egress_allow" {
  security_group_id = aws_security_group.rds_security_group.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1" # semantically equivalent to all ports
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_ec2_security_group_name
  })
}
