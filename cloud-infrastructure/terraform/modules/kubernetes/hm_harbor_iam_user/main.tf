terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user
resource "aws_iam_user" "hm_harbor_iam_user" {
  name = var.aws_iam_user_name
  path = "/hm/"
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.aws_iam_user_name
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user_policy
resource "aws_iam_user_policy" "hm_aws_iam_user_policy" {
  name = "S3ReadWritePolicy-${var.s3_bucket_name}"
  user = aws_iam_user.hm_harbor_iam_user.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })
}
