# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role" "hm_aws_glue_databrew_iam_role" {
  name = "AWSGlueDataBrewServiceRole-${var.source_name}"
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
    Name        = "AWSGlueDataBrewServiceRole-${var.source_name}"
  }
}
resource "aws_iam_role_policy" "hm_aws_glue_databrew_iam_role_input_policy" {
  name = "AwsGlueDataBrewServicePolicyForInputS3-${var.source_name}"
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
          "arn:aws:s3:::${var.input_s3_bucket}/*",
          "arn:aws:s3:::${var.input_s3_bucket}"
        ]
      }
    ]
  })
}
resource "aws_iam_role_policy" "hm_aws_glue_databrew_iam_role_output_policy" {
  name = "AwsGlueDataBrewServicePolicyForOutputS3-${var.source_name}"
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
          "arn:aws:s3:::${var.output_s3_bucket}/*",
          "arn:aws:s3:::${var.output_s3_bucket}"
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
resource "aws_iam_role_policy_attachment" "hm_aws_glue_databrew_iam_role_policy_attachment" {
  role       = aws_iam_role.hm_aws_glue_databrew_iam_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueDataBrewServiceRole"
}

# https://registry.terraform.io/providers/hashicorp/awscc/latest/docs/resources/databrew_job
resource "awscc_databrew_job" "hm_aws_glue_databrew_job" {
  name         = var.aws_glue_databrew_job_name
  role_arn     = aws_iam_role.hm_aws_glue_databrew_iam_role.arn
  type         = "RECIPE"
  dataset_name = "${var.source_name}-data"
  max_capacity = 20
  recipe = {
    name    = "${var.source_name}-recipe"
    version = var.recipe_version
  }
  outputs = [
    {
      location = {
        bucket = var.output_s3_bucket
        key    = var.output_s3_dir
      }
      format           = "PARQUET"
      max_output_files = 1
      overwrite        = true
    }
  ]
  tags = [
    {
      key   = "Environment"
      value = var.environment
    },
    {
      key   = "Team"
      value = var.team
    },
    {
      key   = "Name"
      value = var.aws_glue_databrew_job_name
    }
  ]
}
