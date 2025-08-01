terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "glue_databrew_role" {
  name = "GlueDataBrewServiceRole-${var.aws_glue_databrew_job_nickname}"
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
  tags = merge(var.common_tags, {
    "hm:resource_name" = "AWSGlueDataBrewServiceRole-${var.aws_glue_databrew_job_nickname}"
  })
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "glue_databrew_role_input_policy" {
  name = "AwsGlueDataBrewServicePolicyForInputS3-${var.aws_glue_databrew_job_nickname}"
  role = aws_iam_role.glue_databrew_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.input_s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.input_s3_bucket_name}/*"
        ]
      }
    ]
  })
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "glue_databrew_role_output_policy" {
  name = "AwsGlueDataBrewServicePolicyForOutputS3-${var.aws_glue_databrew_job_nickname}"
  role = aws_iam_role.glue_databrew_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.output_s3_bucket_name}"
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
          "arn:aws:s3:::${var.output_s3_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObjectAcl"
        ]
        Resource = [
          "arn:aws:s3:::${var.output_s3_bucket_name}/*"
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
resource "aws_iam_role_policy_attachment" "glue_databrew_role_policy_attachment" {
  role       = aws_iam_role.glue_databrew_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueDataBrewServiceRole"
}
