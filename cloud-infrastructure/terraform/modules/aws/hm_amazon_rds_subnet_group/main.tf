terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/db_subnet_group
resource "aws_db_subnet_group" "hm_amazon_rds_subnet_group" {
  name       = var.subnet_group_name
  subnet_ids = var.subnet_ids
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.subnet_group_name
  }
}
