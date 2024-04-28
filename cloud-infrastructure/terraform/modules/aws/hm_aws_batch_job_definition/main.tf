terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/batch_job_definition
resource "aws_batch_job_definition" "hm_aws_batch_job_definition" {
  name                  = var.aws_batch_job_definition_name
  type                  = "container"
  platform_capabilities = ["FARGATE"]
  container_properties = jsonencode({
    command = ["echo", "hello"]
    image   = "docker.io/busybox:latest"
    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    }
    resourceRequirements = [
      {
        type  = "VCPU"
        value = "0.25"
      },
      {
        type  = "MEMORY"
        value = "512"
      }
    ]
    executionRoleArn = var.iam_role_arn
  })
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.aws_batch_job_definition_name
  }
}
