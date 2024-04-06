terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_amazon_emr_studio_iam_role" {
  name = "AmazonEMRStudioServiceRole-${var.amazon_emr_studio_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "AmazonEMRStudioServiceRole-${var.amazon_emr_studio_name}"
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_amazon_emr_studio_iam_role_s3_policy" {
  name = "AmazonEMRStudioServicePolicyForS3-${var.amazon_emr_studio_name}"
  role = aws_iam_role.hm_amazon_emr_studio_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetEncryptionConfiguration",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
        Resource = [
          "*"
        ]
      }
    ]
  })
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "hm_amazon_emr_studio_iam_role_policy_attachment" {
  role       = aws_iam_role.hm_amazon_emr_studio_iam_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceEditorsRole"
}
