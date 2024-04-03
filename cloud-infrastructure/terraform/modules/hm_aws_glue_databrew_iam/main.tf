terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_aws_glue_databrew_iam_role" {
  name = "AWSGlueDataBrewServiceRole-${var.aws_glue_databrew_job_nickname}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "databrew.amazonaws.com"
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "AWSGlueDataBrewServiceRole-${var.aws_glue_databrew_job_nickname}"
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_aws_glue_databrew_iam_role_input_policy" {
  name = "AwsGlueDataBrewServicePolicyForInputS3-${var.aws_glue_databrew_job_nickname}"
  role = aws_iam_role.hm_aws_glue_databrew_iam_role.name
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
          "arn:aws:s3:::${var.input_s3_bucket}",
          "arn:aws:s3:::${var.input_s3_bucket}/*"
        ]
      }
    ]
  })
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "hm_aws_glue_databrew_iam_role_output_policy" {
  name = "AwsGlueDataBrewServicePolicyForOutputS3-${var.aws_glue_databrew_job_nickname}"
  role = aws_iam_role.hm_aws_glue_databrew_iam_role.name
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
          "arn:aws:s3:::${var.output_s3_bucket}",
          "arn:aws:s3:::${var.output_s3_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObjectAcl"
        ]
        Resource = [
          "arn:aws:s3:::${var.output_s3_bucket}/*"
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

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "hm_aws_glue_databrew_iam_role_policy_attachment" {
  role       = aws_iam_role.hm_aws_glue_databrew_iam_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueDataBrewServiceRole"
}
