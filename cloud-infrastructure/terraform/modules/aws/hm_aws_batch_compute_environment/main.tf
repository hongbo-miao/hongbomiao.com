terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/batch_compute_environment
resource "aws_batch_compute_environment" "main" {
  name = var.aws_batch_compute_environment_name
  compute_resources {
    max_vcpus          = 256
    security_group_ids = var.amazon_ec2_security_group_ids
    subnets            = var.amazon_vpc_subnet_ids
    type               = "FARGATE"
  }
  service_role = var.iam_role_arn
  type         = "MANAGED"
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.aws_batch_compute_environment_name
  })
}
