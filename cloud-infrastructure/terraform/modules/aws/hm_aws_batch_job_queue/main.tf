terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/batch_job_queue
resource "aws_batch_job_queue" "main" {
  name                 = var.aws_batch_job_queue_name
  state                = "ENABLED"
  priority             = "0"
  compute_environments = var.aws_batch_compute_environment_arns
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.aws_batch_job_queue_name
  }
}
