terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "TempoRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_tempo_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.tempo_service_account_name}"
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.tempo_namespace}:${var.tempo_service_account_name}"
          }
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.tempo_service_account_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_tempo_iam_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.tempo_service_account_name}"
  role = aws_iam_role.hm_tempo_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.tempo_admin_s3_bucket_name}",
          "arn:aws:s3:::${var.tempo_admin_s3_bucket_name}/*",
          "arn:aws:s3:::${var.tempo_trace_s3_bucket_name}",
          "arn:aws:s3:::${var.tempo_trace_s3_bucket_name}/*",
        ]
      }
    ]
  })
}
