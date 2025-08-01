terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group
resource "aws_security_group" "msk_security_group" {
  name   = var.amazon_ec2_security_group_name
  vpc_id = var.amazon_vpc_id
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_ec2_security_group_name
  })
}
# Ingress - On-Site
locals {
  ingress_rule_on_site_kafka_broker_tls        = "on_site_kafka_broker_tls"
  ingress_rule_on_site_kafka_broker_sasl_scram = "on_site_kafka_broker_sasl_scram"
  ingress_rule_on_site_kafka_broker_iam        = "on_site_kafka_broker_iam"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_ingress_rule
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_on_site_kafka_broker_tls" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_on_site_kafka_broker_tls
  cidr_ipv4         = "10.10.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 9094
  to_port           = 9094
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_on_site_kafka_broker_tls
  })
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_on_site_kafka_broker_sasl_scram" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_on_site_kafka_broker_sasl_scram
  cidr_ipv4         = "10.10.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 9096
  to_port           = 9096
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_on_site_kafka_broker_sasl_scram
  })
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_on_site_kafka_broker_iam" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_on_site_kafka_broker_iam
  cidr_ipv4         = "10.10.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 9098
  to_port           = 9098
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_on_site_kafka_broker_iam
  })
}
# Ingress - VPN
locals {
  ingress_rule_vpn_kafka_broker_tls        = "vpn_kafka_broker_tls"
  ingress_rule_vpn_kafka_broker_sasl_scram = "vpn_kafka_broker_sasl_scram"
  ingress_rule_vpn_kafka_broker_iam        = "vpn_kafka_broker_iam"
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpn_kafka_broker_tls" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_vpn_kafka_broker_tls
  cidr_ipv4         = "10.100.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 9094
  to_port           = 9094
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpn_kafka_broker_tls
  })
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpn_kafka_broker_sasl_scram" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_vpn_kafka_broker_sasl_scram
  cidr_ipv4         = "10.100.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 9096
  to_port           = 9096
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpn_kafka_broker_sasl_scram
  })
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpn_kafka_broker_iam" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_vpn_kafka_broker_iam
  cidr_ipv4         = "10.100.0.0/15"
  ip_protocol       = "tcp"
  from_port         = 9098
  to_port           = 9098
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpn_kafka_broker_iam
  })
}
# Ingress - VPC
locals {
  ingress_rule_vpc_kafka_broker_tls        = "vpc_kafka_broker_tls"
  ingress_rule_vpc_kafka_broker_sasl_scram = "vpc_kafka_broker_sasl_scram"
  ingress_rule_vpc_kafka_broker_iam        = "vpc_kafka_broker_iam"
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpc_kafka_broker_tls" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_vpc_kafka_broker_tls
  cidr_ipv4         = var.amazon_vpc_cidr_ipv4
  ip_protocol       = "tcp"
  from_port         = 9094
  to_port           = 9094
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpc_kafka_broker_tls
  })
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpc_kafka_broker_sasl_scram" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_vpc_kafka_broker_sasl_scram
  cidr_ipv4         = var.amazon_vpc_cidr_ipv4
  ip_protocol       = "tcp"
  from_port         = 9096
  to_port           = 9096
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpc_kafka_broker_sasl_scram
  })
}
resource "aws_vpc_security_group_ingress_rule" "ingress_rule_vpc_kafka_broker_iam" {
  security_group_id = aws_security_group.msk_security_group.id
  description       = local.ingress_rule_vpc_kafka_broker_iam
  cidr_ipv4         = var.amazon_vpc_cidr_ipv4
  ip_protocol       = "tcp"
  from_port         = 9098
  to_port           = 9098
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.ingress_rule_vpc_kafka_broker_iam
  })
}
# Egress
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/vpc_security_group_egress_rule
resource "aws_vpc_security_group_egress_rule" "egress_allow" {
  security_group_id = aws_security_group.msk_security_group.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1" # semantically equivalent to all ports
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_ec2_security_group_name
  })
}
