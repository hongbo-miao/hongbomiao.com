terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group
resource "aws_security_group" "hm_airbyte_postgres_security_group" {
  name   = var.amazon_ec2_security_group_name
  vpc_id = var.amazon_vpc_id
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.amazon_ec2_security_group_name
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_ingress_rule
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_on_site" {
  security_group_id = aws_security_group.hm_airbyte_postgres_security_group.id
  description       = "On-Site"
  cidr_ipv4         = "10.10.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 5432
  to_port           = 5432
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "On-Site"
  }
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpn" {
  security_group_id = aws_security_group.hm_airbyte_postgres_security_group.id
  description       = "VPN"
  cidr_ipv4         = "10.100.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 5432
  to_port           = 5432
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "VPN"
  }
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpc" {
  security_group_id = aws_security_group.hm_airbyte_postgres_security_group.id
  description       = "VPC"
  cidr_ipv4         = "172.16.0.0/12"
  ip_protocol       = "tcp"
  from_port         = 5432
  to_port           = 5432
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "VPC"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_egress_rule
resource "aws_vpc_security_group_egress_rule" "egress_allow" {
  security_group_id = aws_security_group.hm_airbyte_postgres_security_group.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1" # semantically equivalent to all ports
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.amazon_ec2_security_group_name
  }
}
