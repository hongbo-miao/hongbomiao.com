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
  client_authentication {
    sasl {
      iam = true
    }
  }
  broker_node_group_info {
    instance_type = var.instance_type
    security_groups = [
      "sg-xxxxxxxxxxxxxxxxx"
    ]
    client_subnets = [
      "subnet-xxxxxxxxxxxxxxxxx",
      "subnet-xxxxxxxxxxxxxxxxx"
    ]
  }
  encryption_info {
    encryption_at_rest_kms_key_arn = "arn:aws:kms:us-west-2:272394222652:key/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  }
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.amazon_msk_cluster_name
  }
}
