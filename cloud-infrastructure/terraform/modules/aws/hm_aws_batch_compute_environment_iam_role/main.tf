terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "batch_compute_environment_role" {
  name = "BatchComputeEnvironmentServiceRole-${var.aws_batch_compute_environment_nickname}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "batch.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  tags = merge(var.common_tags, {
    "hm:resource_name" = "AWSBatchComputeEnvironmentServiceRole-${var.aws_batch_compute_environment_nickname}"
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "batch_compute_environment_role_policy_attachment" {
  role       = aws_iam_role.batch_compute_environment_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}
