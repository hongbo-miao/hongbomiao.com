terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_studio
resource "aws_emr_studio" "hm_amazon_emr_studio" {
  name                        = var.amazon_emr_studio_name
  auth_mode                   = "IAM"
  default_s3_location         = var.s3_uri
  service_role                = var.iam_role_arn
  engine_security_group_id    = "sg-xxxxxxxxxxxxxxxxx"
  workspace_security_group_id = "sg-xxxxxxxxxxxxxxxxx"
  vpc_id                      = "vpc-xxxxxxxxxxxxxxxxx"
  subnet_ids                  = ["subnet-xxxxxxxxxxxxxxxxx"]
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.amazon_emr_studio_name
  }
}
