terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/sagemaker_notebook_instance
resource "aws_sagemaker_notebook_instance" "hm_amazon_sagemaker_notebook_instance" {
  name          = var.amazon_sagemaker_notebook_instance_name
  role_arn      = var.iam_role_arn
  instance_type = var.instance_type
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.amazon_sagemaker_notebook_instance_name
  }
}
