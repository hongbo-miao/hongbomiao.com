terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://docs.aws.amazon.com/msk/latest/developerguide/msk-connect-service-execution-role.html
locals {
  aws_iam_role_name_prefix = "MSKConnectorRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "msk_s3_sink_connector_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.amazon_msk_connector_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "kafkaconnect.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "${local.aws_iam_role_name_prefix}-${var.amazon_msk_connector_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_amazon_msk_connector_iam_role_msk_policy" {
  name = "${local.aws_iam_role_name_prefix}MSKPolicy-${var.amazon_msk_connector_name}"
  role = aws_iam_role.msk_s3_sink_connector_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:DescribeCluster"
        ]
        Resource = [
          var.amazon_msk_arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:AlterGroup",
          "kafka-cluster:DescribeGroup"
        ]
        Resource = [
          "${replace(var.amazon_msk_arn, ":cluster", ":group")}/__amazon_msk_connect_*",
          "${replace(var.amazon_msk_arn, ":cluster", ":group")}/connect-*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:CreateTopic",
          "kafka-cluster:ReadData",
          "kafka-cluster:WriteData"
        ]
        Resource = [
          "${replace(var.amazon_msk_arn, ":cluster", ":topic")}/__amazon_msk_connect_*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:ReadData"
        ]
        Resource = [
          "${replace(var.amazon_msk_arn, ":cluster", ":topic")}/*"
        ]
      }
    ]
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_amazon_msk_connector_iam_role_plugin_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}PluginS3Policy-${var.amazon_msk_connector_name}"
  role = aws_iam_role.msk_s3_sink_connector_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_plugin_s3_bucket_name}",
          "arn:aws:s3:::${var.msk_plugin_s3_bucket_name}/*"
        ]
      }
    ]
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_amazon_msk_connector_iam_role_log_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}LogS3Policy-${var.amazon_msk_connector_name}"
  role = aws_iam_role.msk_s3_sink_connector_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_log_s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_log_s3_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObjectAcl"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_log_s3_bucket_name}/*"
        ],
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      }
    ]
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_amazon_msk_connector_iam_role_data_s3_policy" {
  name = "${local.aws_iam_role_name_prefix}DataS3Policy-${var.amazon_msk_connector_name}"
  role = aws_iam_role.msk_s3_sink_connector_role.name
  # https://docs.confluent.io/kafka-connectors/s3-sink/current/overview.html#iam-policy-for-s3
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
        Resource = [
          "arn:aws:s3::*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_data_s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:GetObject",
          "s3:PutObject",
          "s3:PutObjectTagging"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_data_s3_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObjectAcl"
        ]
        Resource = [
          "arn:aws:s3:::${var.msk_data_s3_bucket_name}/*"
        ],
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      }
    ]
  })
}
