terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
# https://docs.aws.amazon.com/msk/latest/developerguide/msk-connect-service-execution-role.html
resource "aws_iam_role" "hm_amazon_msk_connector_iam_role" {
  name = "AmazonMSKConnectorServiceRole-${var.amazon_msk_connector_name}"
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
    ResourceName = "AmazonMSKConnectorServiceRole-${var.amazon_msk_connector_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_amazon_msk_connector_iam_role_policy" {
  name = "AmazonMSKConnectorServicePolicyForMSK-${var.amazon_msk_connector_name}"
  role = aws_iam_role.hm_amazon_msk_connector_iam_role.name
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
          "*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:WriteData"
        ]
        Resource = [
          "*"
        ]
      }
    ]
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_aws_glue_databrew_iam_role_input_policy" {
  name = "AmazonMSKConnectorServicePolicyForMSKPluginS3-${var.amazon_msk_connector_name}"
  role = aws_iam_role.hm_amazon_msk_connector_iam_role.name
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
resource "aws_iam_role_policy" "hm_aws_glue_databrew_iam_role_kafka_worker_log_s3_policy" {
  name = "AmazonMSKConnectorServicePolicyForMSKLogS3-${var.amazon_msk_connector_name}"
  role = aws_iam_role.hm_amazon_msk_connector_iam_role.name
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
          "arn:aws:s3:::${var.msk_log_s3_bucket_name}",
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
