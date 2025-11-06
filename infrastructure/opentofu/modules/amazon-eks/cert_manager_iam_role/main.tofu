terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_prefix      = "CertManagerRole"
  aws_iam_role_name        = "${local.aws_iam_role_prefix}-${var.cert_manager_service_account_name}-${var.amazon_eks_cluster_name_hash}"
  aws_iam_role_policy_name = "${local.aws_iam_role_prefix}Policy-${var.cert_manager_service_account_name}-${var.amazon_eks_cluster_name_hash}"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "cert_manager_role" {
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.cert_manager_namespace}:${var.cert_manager_service_account_name}"
          }
        }
      }
    ]
  })
  tags = merge(var.common_tags, {
    "hm:resource_name" = local.aws_iam_role_name
  })
}
# https://cert-manager.io/docs/configuration/acme/dns01/route53
# https://registry.terraform.io/providers/vancluever/acme/latest/docs/guides/dns-providers-route53#least-privilege-policy-for-production-purposes
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "cert_manager_role_policy" {
  name = local.aws_iam_role_policy_name
  role = aws_iam_role.cert_manager_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "route53:GetChange"
        ]
        Resource = "arn:aws:route53:::change/*"
      },
      {
        Effect = "Allow"
        Action = [
          "route53:ListResourceRecordSets"
        ]
        Resource = "arn:aws:route53:::hostedzone/${var.amazon_route53_hosted_zone_id}"
      },
      {
        Effect = "Allow"
        Action = [
          "route53:ChangeResourceRecordSets"
        ]
        Resource = "arn:aws:route53:::hostedzone/${var.amazon_route53_hosted_zone_id}"
        Condition = {
          "ForAllValues:StringLike" = {
            "route53:ChangeResourceRecordSetsNormalizedRecordNames" = [
              "_acme-challenge.*"
            ],
            "route53:ChangeResourceRecordSetsRecordTypes" = [
              "TXT"
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "route53:ListHostedZonesByName"
        ]
        Resource = "*"
      }
    ]
  })
}
