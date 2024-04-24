terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_aws_batch_job_definition_iam" {
  name = "AWSBatchJobDefininationExecutionRole-${var.aws_batch_job_definition_nickname}"
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
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "AWSBatchJobDefininationExecutionRole-${var.aws_batch_job_definition_nickname}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "hm_aws_batch_job_definition_iam_policy_attachment" {
  role       = aws_iam_role.hm_aws_batch_job_definition_iam.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
