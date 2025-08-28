terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_prefix      = "ExternalDNSRole"
  aws_iam_role_name        = "${local.aws_iam_role_prefix}-${var.external_dns_service_account_name}-${var.amazon_eks_cluster_name_hash}"
  aws_iam_role_policy_name = "${local.aws_iam_role_prefix}Policy-${var.external_dns_service_account_name}-${var.amazon_eks_cluster_name_hash}"
}
# https://github.com/kubernetes-sigs/external-dns/blob/master/docs/tutorials/aws.md
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "external_dns_role" {
  name = local.aws_iam_role_name
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
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.aws_iam_role_name
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
# https://github.com/kubernetes-sigs/external-dns/blob/master/docs/tutorials/aws.md
resource "aws_iam_role_policy" "external_dns_role_policy" {
  name = local.aws_iam_role_policy_name
  role = aws_iam_role.external_dns_role.name
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
