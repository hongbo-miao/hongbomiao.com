terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/sagemaker_notebook_instance
resource "aws_sagemaker_notebook_instance" "main" {
  name          = var.amazon_sagemaker_notebook_instance_name
  role_arn      = var.iam_role_arn
  instance_type = var.instance_type
  tags = merge(var.common_tags, {
    "hm_resource_name" = var.amazon_sagemaker_notebook_instance_name
  })
}
