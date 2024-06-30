terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "CertManagerRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user
resource "aws_iam_role" "hm_cert_manager_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.cert_manager_service_account_name}"
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
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.cert_manager_service_account_name}"
  }
}
# https://cert-manager.io/docs/configuration/acme/dns01/route53
# https://registry.terraform.io/providers/vancluever/acme/latest/docs/guides/dns-providers-route53#least-privilege-policy-for-production-purposes
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user_policy
resource "aws_iam_role_policy" "hm_cert_manager_iam_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.cert_manager_service_account_name}"
  role = aws_iam_role.hm_cert_manager_iam_role.name
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
        Resource = "arn:aws:route53:::hostedzone/${var.amazon_route53_hosted_zone_id}",
        "Condition" : {
          "ForAllValues:StringEquals" : {
            "route53:ChangeResourceRecordSetsNormalizedRecordNames" : [
              "_acme-challenge.*"
            ],
            "route53:ChangeResourceRecordSetsRecordTypes" : [
              "TXT"
            ]
          }
        }
      },
      {
        "Effect" = "Allow",
        "Action" = [
          "route53:ListHostedZonesByName"
        ],
        "Resource" = "*"
      }
    ]
  })
}
