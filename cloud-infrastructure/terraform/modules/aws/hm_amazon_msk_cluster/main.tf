terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/msk_cluster
resource "aws_msk_cluster" "hm_amazon_msk_cluster" {
  cluster_name           = var.amazon_msk_cluster_name
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.kafka_broker_number
  storage_mode           = "TIERED"
  broker_node_group_info {
    instance_type   = var.kafka_broker_instance_type
    security_groups = [var.amazon_vpc_security_group_id]
    client_subnets  = var.amazon_vpc_subnet_ids
    storage_info {
      ebs_storage_info {
        volume_size = var.amazon_ebs_volume_size_gb
      }
    }
  }
  logging_info {
    broker_logs {
      s3 {
        enabled = true
        bucket  = var.kafka_broker_log_s3_bucket_name
        prefix  = "brokers"
      }
    }
  }
  encryption_info {
    encryption_at_rest_kms_key_arn = var.aws_kms_key_arn
  }
  client_authentication {
    sasl {
      iam = true
    }
  }
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.amazon_msk_cluster_name
  }
}
