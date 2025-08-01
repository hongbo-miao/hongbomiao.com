terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_prefix = "EKSAccessEntryRole"
  aws_iam_role_name   = "${local.aws_iam_role_prefix}-${var.amazon_eks_access_entry_name}-${var.amazon_eks_cluster_name_hash}"
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "eks_access_entry_role" {
  name = local.aws_iam_role_name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  tags = merge(var.common_tags, {
    "hm:resource_name" = "${local.aws_iam_role_prefix}-${var.amazon_eks_access_entry_name}"
  })
}
