terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "RedpandaConsoleRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user
resource "aws_iam_role" "hm_redpanda_console_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.redpanda_console_service_account_name}"
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.redpanda_console_namespace}:${var.redpanda_console_service_account_name}"
          }
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.redpanda_console_service_account_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user_policy
resource "aws_iam_role_policy" "hm_redpanda_console_iam_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.redpanda_console_service_account_name}"
  role = aws_iam_role.hm_redpanda_console_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:DescribeCluster",
          "kafka-cluster:DescribeClusterDynamicConfiguration",
          "kafka-cluster:DescribeGroup",
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:DescribeTopicDynamicConfiguration",
          "kafka-cluster:DescribeTransactionalId",
          "kafka-cluster:ReadData"
        ]
        Resource = [
          "*"
        ]
      }
    ]
  })
}
