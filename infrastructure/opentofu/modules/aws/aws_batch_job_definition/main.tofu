terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/batch_job_definition
resource "aws_batch_job_definition" "main" {
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
  tags = merge(var.common_tags, {
    "hm_resource_name" = var.aws_batch_job_definition_name
  })
}
