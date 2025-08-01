terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "SchemaRegistryRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "confluent_schema_registry_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.confluent_schema_registry_service_account_nickname}"
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.confluent_schema_registry_namespace}:${var.confluent_schema_registry_service_account_name}"
          }
        }
      }
    ]
  })
  tags = merge(var.common_tags, {
    "hm:resource_name" = "${local.aws_iam_role_name_prefix}-${var.confluent_schema_registry_service_account_name}"
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "confluent_schema_registry_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.confluent_schema_registry_service_account_name}"
  role = aws_iam_role.confluent_schema_registry_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:DescribeCluster",
          "kafka-cluster:DescribeClusterDynamicConfiguration"
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
          "${replace(var.amazon_msk_arn, ":cluster", ":group")}/schema-registry"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:AlterTopic",
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:DescribeTopicDynamicConfiguration",
          "kafka-cluster:CreateTopic",
          "kafka-cluster:ReadData",
          "kafka-cluster:WriteData"
        ]
        Resource = [
          "${replace(var.amazon_msk_arn, ":cluster", ":topic")}/_schemas"
        ]
      }
    ]
  })
}
