terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "batch_job_definition_role" {
  name = "BatchJobDefininationExecutionRole-${var.aws_batch_job_definition_nickname}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  tags = merge(var.common_tags, {
    "hm_resource_name" = "AWSBatchJobDefininationExecutionRole-${var.aws_batch_job_definition_nickname}"
  })
}
# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "batch_job_definition_role_policy_attachment" {
  role       = aws_iam_role.batch_job_definition_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
