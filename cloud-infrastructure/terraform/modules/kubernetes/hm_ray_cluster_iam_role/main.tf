terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "RayClusterRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "ray_cluster_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.ray_cluster_service_account_name}"
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.ray_cluster_namespace}:${var.ray_cluster_service_account_name}"
          }
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.ray_cluster_service_account_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "mlflow_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}MLflowS3Policy-${var.ray_cluster_service_account_name}"
  role = aws_iam_role.ray_cluster_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.mlflow_s3_bucket_name}/*"
        ]
      }
    ]
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "iot_data_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}IotDataS3Policy-${var.ray_cluster_service_account_name}"
  role = aws_iam_role.ray_cluster_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.iot_data_s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.iot_data_s3_bucket_name}/*"
        ]
      }
    ]
  })
}
