terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/batch_job_queue
resource "aws_batch_job_queue" "main" {
  name     = var.aws_batch_job_queue_name
  state    = "ENABLED"
  priority = "0"
  dynamic "compute_environment_order" {
    for_each = var.aws_batch_compute_environment_arns
    content {
      order               = compute_environment_order.key + 1
      compute_environment = compute_environment_order.value
    }
  }
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.aws_batch_job_queue_name
  })
}
