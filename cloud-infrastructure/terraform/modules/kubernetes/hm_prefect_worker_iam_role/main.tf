terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  aws_iam_role_name_prefix = "PrefectWorkerRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "prefect_worker_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.prefect_worker_service_account_name}"
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.prefect_worker_namespace}:${var.prefect_worker_service_account_name}"
          }
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.prefect_worker_service_account_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "iot_data_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}IotDataS3Policy-${var.prefect_worker_service_account_name}"
  role = aws_iam_role.prefect_worker_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
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
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "aws_glue_policy" {
  name = "${local.aws_iam_role_name_prefix}AwsGluePolicy-${var.prefect_worker_service_account_name}"
  role = aws_iam_role.prefect_worker_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables"
        ]
        Resource = flatten([
          "arn:aws:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:catalog",
          [for database in var.aws_glue_database_names :
            "arn:aws:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${database}"
          ],
          [for database in var.aws_glue_database_names :
            "arn:aws:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${database}/*"
          ]
        ])
      }
    ]
  })
}
