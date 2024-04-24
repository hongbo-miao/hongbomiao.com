terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/batch_job_queue
resource "aws_batch_job_queue" "hm_aws_batch_job_queue" {
  name                 = var.aws_batch_job_queue_name
  state                = "ENABLED"
  priority             = "0"
  compute_environments = var.aws_batch_compute_environment_arns
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.aws_batch_job_queue_name
  }
}
