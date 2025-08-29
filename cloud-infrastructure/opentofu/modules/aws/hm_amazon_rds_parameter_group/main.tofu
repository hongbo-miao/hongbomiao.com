terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/db_parameter_group
resource "aws_db_parameter_group" "rds_parameter_group" {
  name   = var.parameter_group_name
  family = var.family
  parameter {
    name  = "log_connections"
    value = "1"
  }
  dynamic "parameter" {
    for_each = var.parameters
    content {
      name  = parameter.value.name
      value = parameter.value.value
    }
  }
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.parameter_group_name
  })
  lifecycle {
    create_before_destroy = true
  }
}
