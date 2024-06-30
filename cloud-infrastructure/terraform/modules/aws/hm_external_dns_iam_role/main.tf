terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "ExternalDNSRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
# https://github.com/kubernetes-sigs/external-dns/blob/master/docs/tutorials/aws.md
resource "aws_iam_role" "hm_external_dns_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.external_dns_service_account_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.amazon_eks_cluster_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.amazon_eks_cluster_oidc_provider}:aud" = "sts.amazonaws.com",
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.external_dns_namespace}:${var.external_dns_service_account_name}"
          }
        }
      }
    ]
  })
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "${local.aws_iam_role_name_prefix}-${var.external_dns_service_account_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
# https://github.com/kubernetes-sigs/external-dns/blob/master/docs/tutorials/aws.md
resource "aws_iam_role_policy" "hm_external_dns_iam_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.external_dns_service_account_name}"
  role = aws_iam_role.hm_external_dns_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "route53:ChangeResourceRecordSets"
        ]
        Resource = [
          "arn:aws:route53:::hostedzone/${var.amazon_route53_hosted_zone_id}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "route53:ListHostedZones",
          "route53:ListResourceRecordSets",
          "route53:ListTagsForResource"
        ]
        Resource = [
          "*"
        ]
      }
    ]
  })
}
