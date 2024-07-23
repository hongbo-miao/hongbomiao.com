terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  name_prefix                          = "KafkaManager"
  aws_iam_instance_profile_name_prefix = "${local.name_prefix}InstanceProfile"
  aws_iam_role_name_prefix             = "${local.name_prefix}Role"
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_instance_profile
resource "aws_iam_instance_profile" "hm_kafka_manager_ec2_instance_profile" {
  name = "${local.aws_iam_instance_profile_name_prefix}-${var.kafka_manager_name}"
  role = aws_iam_role.hm_kafka_manager_iam_role.name
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_kafka_manager_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.kafka_manager_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.kafka_manager_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_kafka_manager_iam_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.kafka_manager_name}"
  role = aws_iam_role.hm_kafka_manager_iam_role.name
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
          "${replace(var.amazon_msk_arn, ":cluster", ":group")}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:CreateTopic",
          "kafka-cluster:ReadData",
          "kafka-cluster:WriteData",
          "kafka-cluster:AlterTopic",
          "kafka-cluster:AlterTopicDynamicConfiguration",
          "kafka-cluster:DeleteTopic"
        ]
        Resource = [
          "${replace(var.amazon_msk_arn, ":cluster", ":topic")}/*"
        ]
      }
    ]
  })
}
