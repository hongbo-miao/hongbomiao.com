terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "LiteLLMRole"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_litellm_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.litellm_service_account_name}"
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
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:${var.litellm_namespace}:${var.litellm_service_account_name}"
          }
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "${local.aws_iam_role_name_prefix}-${var.litellm_service_account_name}"
  }
}
# https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html#security_iam_id-based-policy-examples-perform-actions-pt
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
# https://registry.terraform.io/providers/vancluever/acme/latest/docs/guides/dns-providers-route53#least-privilege-policy-for-production-purposes
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_litellm_iam_role_policy" {
  name = "${local.aws_iam_role_name_prefix}Policy-${var.litellm_service_account_name}"
  role = aws_iam_role.hm_litellm_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          # Claude Haiku
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0",
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
          # Claude Opus
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-opus-20240229-v1:0",
          # Claude Sonnet
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
        ]
      }
    ]
  })
}
