terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "LabelStudioRole"
}
# https://docs.labelbox.com/docs/import-aws-s3-data
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "labelbox_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.labelbox_service_account_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.labelbox_aws_account_id}:role/lb-aws-delegated-access-role"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "sts:ExternalId" = var.external_id
          }
        }
      }
    ]
  })
  tags = merge(var.common_tags, {
    "hm:resource_name" = "${local.aws_iam_role_name_prefix}-${var.labelbox_service_account_name}"
  })
}
# https://docs.labelbox.com/docs/import-aws-s3-data
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "labelbox_iam_role_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}S3Policy-${var.labelbox_service_account_name}"
  role = aws_iam_role.labelbox_iam_role.name
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
          "arn:aws:s3:::${var.s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })
}
