terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_amazon_eks_access_entry_iam" {
  name = "AmazonEKSAcessEntryRole-${var.amazon_eks_access_entry_name}"
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
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "AmazonEKSAcessEntryRole-${var.amazon_eks_access_entry_name}"
  }
}
