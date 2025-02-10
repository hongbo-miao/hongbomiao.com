terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/db_instance
resource "aws_db_instance" "rds_instance" {
  identifier                            = var.amazon_rds_name
  engine                                = var.amazon_rds_engine
  engine_version                        = var.amazon_rds_engine_version
  instance_class                        = var.amazon_rds_instance_class
  allocated_storage                     = var.storage_size_gb
  max_allocated_storage                 = var.max_storage_size_gb
  storage_type                          = "gp3"
  username                              = var.user_name
  password                              = var.password
  vpc_security_group_ids                = var.vpc_security_group_ids
  db_subnet_group_name                  = var.subnet_group_name
  parameter_group_name                  = var.parameter_group_name
  publicly_accessible                   = false
  storage_encrypted                     = true
  ca_cert_identifier                    = "rds-ca-ecc384-g1"
  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  enabled_cloudwatch_logs_exports       = var.cloudwatch_log_types
  skip_final_snapshot                   = false
  backup_retention_period               = 7
  deletion_protection                   = true
  apply_immediately                     = true
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.amazon_rds_name
  }
  lifecycle {
    prevent_destroy = true
  }
}
